import argparse
import csv
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode
from act_kat.ollama_client import ollama_generate
from act_kat.prompting import build_chat_user_message
from act_kat.replay import replay_actions, upsample_waypoints_linear
from act_kat.vision_tokens import tokenize_keypoints_2d
from sim_env import BOX_POSE


def build_schema(M: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": M,
                "maxItems": M,
                "items": {
                    "type": "object",
                    "properties": {
                        "L": {
                            "type": "array",
                            "minItems": 6,
                            "maxItems": 6,
                            "items": {"type": "integer"},
                        },
                        "LG": {"type": "integer", "enum": [0, 1]},
                        "R": {
                            "type": "array",
                            "minItems": 6,
                            "maxItems": 6,
                            "items": {"type": "integer"},
                        },
                        "RG": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["L", "LG", "R", "RG"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemma4:26b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--out_csv", type=str, default="artifacts/results_icl_json.csv")
    ap.add_argument("--n_eval", type=int, default=5)
    ap.add_argument("--n_demos", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--demo_ids",
        type=str,
        default=None,
        help="Optional comma-separated fixed demo episode ids (overrides random demo sampling).",
    )
    ap.add_argument(
        "--query_ids",
        type=str,
        default=None,
        help="Optional comma-separated fixed query episode ids (overrides random query sampling).",
    )
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs("artifacts/replays", exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Discover how many episodes exist in dataset_dir
    all_eps = []
    for name in sorted(os.listdir(args.dataset_dir)):
        if name.startswith("episode_") and name.endswith(".hdf5"):
            try:
                all_eps.append(int(name[len("episode_") : -len(".hdf5")]))
            except Exception:
                pass
    all_eps = sorted(all_eps)
    if not all_eps:
        raise SystemExit(f"No episode_*.hdf5 found in {args.dataset_dir}")

    # Choose fixed demos/queries or sample without leakage
    if args.demo_ids is not None:
        demo_ids = [int(x) for x in args.demo_ids.split(",") if x.strip() != ""]
    else:
        demo_ids = rng.choice(all_eps, size=min(args.n_demos, len(all_eps)), replace=False).tolist()

    remaining = [e for e in all_eps if e not in set(demo_ids)]
    if args.query_ids is not None:
        query_ids = [int(x) for x in args.query_ids.split(",") if x.strip() != ""]
    else:
        if len(remaining) == 0:
            raise SystemExit("No held-out episodes left for queries after selecting demos.")
        n_eval = min(args.n_eval, len(remaining))
        query_ids = rng.choice(remaining, size=n_eval, replace=False).tolist()

    # Build a single quantizer across demos + queries (mins/maxs)
    actions_all = []
    for i in sorted(set(demo_ids + query_ids)):
        with h5py.File(os.path.join(args.dataset_dir, f"episode_{i}.hdf5"), "r") as root:
            actions_all.append(root["/action"][()].astype(np.float32))
    actions_all = np.concatenate(actions_all, axis=0)
    quant = ActionQuantizer(
        mins=actions_all.min(axis=0), maxs=actions_all.max(axis=0), bins=args.bins
    )

    schema = build_schema(args.M)

    rows = []
    successes = 0
    parse_errors = 0
    gen_times = []

    # Build demo pairs once (OBS->ACT) and reuse for all queries
    demo_pairs = []
    for d in demo_ids:
        with h5py.File(os.path.join(args.dataset_dir, f"episode_{d}.hdf5"), "r") as root:
            img0 = root["/observations/images/top"][0]
            act = root["/action"][()].astype(np.float32)
        _, obs_tok = tokenize_keypoints_2d(img0, k=args.K, device=args.device)
        _, act_tok = action_tokens_from_episode(act, quantizer=quant, M=args.M)
        demo_pairs.append(obs_tok + act_tok)

    for idx, query_id in enumerate(query_ids):
        with h5py.File(os.path.join(args.dataset_dir, f"episode_{query_id}.hdf5"), "r") as root:
            q_img0 = root["/observations/images/top"][0]
            env0 = np.array(root.attrs["env_state0"], dtype=np.float32)

        _, query_obs = tokenize_keypoints_2d(q_img0, k=args.K, device=args.device)
        prompt = build_chat_user_message(demo_pairs, query_obs=query_obs) + (
            "\n\nReturn ONLY JSON matching the provided schema.\n"
            f"Output exactly M={args.M} action objects.\n"
        )

        t0 = time.time()
        resp = ollama_generate(
            prompt=prompt,
            model=args.model,
            url=args.ollama_url,
            options={"temperature": 0.0, "num_predict": 2048},
            run_id=f"eval_icl_json_q{query_id}_idx{idx}",
            artifacts_dir="artifacts",
            format=schema,
            timeout_s=600,
        )
        gen_times.append(time.time() - t0)

        text = resp.get("response", "") if isinstance(resp, dict) else ""
        try:
            data = json.loads(text)
            actions = data["actions"]
            if not (isinstance(actions, list) and len(actions) == args.M):
                raise ValueError("actions wrong length")
        except Exception as e:
            parse_errors += 1
            rows.append(
                {
                    "episode_id": query_id,
                    "success": 0,
                    "max_reward": 0.0,
                    "parse_ok": 0,
                    "parse_errors": json.dumps([f"json_parse:{type(e).__name__}"]),
                }
            )
            continue

        codes = np.zeros((args.M, 14), dtype=np.int32)
        for i, a in enumerate(actions):
            L = a["L"]
            R = a["R"]
            LG = int(a["LG"])
            RG = int(a["RG"])
            codes[i, :6] = np.asarray([int(x) for x in L], dtype=np.int32)
            codes[i, 7:13] = np.asarray([int(x) for x in R], dtype=np.int32)
            codes[i, 6] = LG * (args.bins - 1)
            codes[i, 13] = RG * (args.bins - 1)

        actions_wp = quant.decode(codes).astype(np.float32)
        actions_full = upsample_waypoints_linear(actions_wp, T=args.episode_len)

        BOX_POSE[0] = env0
        res = replay_actions(
            actions_full,
            episode_len=args.episode_len,
            video_path=f"artifacts/replays/eval_icl_json_q{query_id}_idx{idx}.mp4",
        )

        successes += int(res.success)
        rows.append(
            {
                "episode_id": query_id,
                "success": int(res.success),
                "max_reward": res.max_reward,
                "parse_ok": 1,
                "parse_errors": "[]",
            }
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode_id", "success", "max_reward", "parse_ok", "parse_errors"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "dataset_dir": args.dataset_dir,
        "demo_ids": demo_ids,
        "query_ids": query_ids,
        "K_keypoints": args.K,
        "M_actions": args.M,
        "success_rate": successes / max(1, args.n_eval),
        "parse_error_rate": parse_errors / max(1, args.n_eval),
        "avg_gen_time_s": float(np.mean(gen_times)) if gen_times else None,
        "out_csv": args.out_csv,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

