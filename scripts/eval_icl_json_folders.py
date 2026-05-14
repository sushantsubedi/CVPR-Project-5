import argparse
import csv
import json
import os
import sys
import time
from typing import List, Tuple

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode
from act_kat.ollama_client import ollama_generate
from act_kat.prompting import build_chat_user_message
from act_kat.replay import replay_actions, upsample_waypoints_piecewise_constant
from act_kat.vision_tokens import tokenize_keypoints_2d, tokenize_keypoints_2d_with_depth
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


def list_eps(dataset_dir: str) -> List[int]:
    out = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.startswith("episode_") and name.endswith(".hdf5"):
            try:
                out.append(int(name[len("episode_") : -len(".hdf5")]))
            except Exception:
                pass
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos_dir", type=str, required=True)
    ap.add_argument("--tests_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemma4:26b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--obs_bins", type=int, default=64)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_csv", type=str, default="artifacts/results_icl_json.csv")
    ap.add_argument("--max_tests", type=int, default=None, help="Optionally cap number of test episodes")
    args = ap.parse_args()

    demo_ids = list_eps(args.demos_dir)
    test_ids = list_eps(args.tests_dir)
    if args.max_tests is not None:
        test_ids = test_ids[: int(args.max_tests)]

    if not demo_ids:
        raise SystemExit("No demos found")
    if not test_ids:
        raise SystemExit("No tests found")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs("artifacts/replays", exist_ok=True)

    # Quantizer from demos only (more realistic)
    actions_all = []
    for ep in demo_ids:
        with h5py.File(os.path.join(args.demos_dir, f"episode_{ep}.hdf5"), "r") as root:
            actions_all.append(root["/action"][()].astype(np.float32))
    actions_all = np.concatenate(actions_all, axis=0)
    quant = ActionQuantizer(
        mins=actions_all.min(axis=0), maxs=actions_all.max(axis=0), bins=args.bins
    )

    schema = build_schema(args.M)

    # Build demo pairs once
    demo_pairs = []
    for ep in demo_ids:
        with h5py.File(os.path.join(args.demos_dir, f"episode_{ep}.hdf5"), "r") as root:
            img0 = root["/observations/images/top"][0]
            d0 = root["/observations/depths/top"][0] if "/observations/depths/top" in root else None
            act = root["/action"][()].astype(np.float32)
        if d0 is not None and np.isfinite(np.asarray(d0)).any():
            _, obs_tok = tokenize_keypoints_2d_with_depth(
                img0, depth=d0, k=args.K, device=args.device, obs_bins=args.obs_bins
            )
        else:
            _, obs_tok = tokenize_keypoints_2d(img0, k=args.K, device=args.device, obs_bins=args.obs_bins)
        _, act_tok = action_tokens_from_episode(act, quantizer=quant, M=args.M)
        demo_pairs.append(obs_tok + act_tok)

    rows = []
    successes = 0
    parse_errors = 0
    gen_times = []

    for idx, ep in enumerate(test_ids):
        with h5py.File(os.path.join(args.tests_dir, f"episode_{ep}.hdf5"), "r") as root:
            q_img0 = root["/observations/images/top"][0]
            q_d0 = root["/observations/depths/top"][0] if "/observations/depths/top" in root else None
            env0 = np.array(root.attrs["env_state0"], dtype=np.float32)

        if q_d0 is not None and np.isfinite(np.asarray(q_d0)).any():
            _, query_obs = tokenize_keypoints_2d_with_depth(
                q_img0, depth=q_d0, k=args.K, device=args.device, obs_bins=args.obs_bins
            )
        else:
            _, query_obs = tokenize_keypoints_2d(q_img0, k=args.K, device=args.device, obs_bins=args.obs_bins)
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
            run_id=f"eval_icl_json_demo{len(demo_ids)}_test{ep}_idx{idx}",
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
                    "episode_id": ep,
                    "success": 0,
                    "max_reward": 0.0,
                    "parse_ok": 0,
                    "parse_errors": json.dumps([f"json_parse:{type(e).__name__}"]),
                }
            )
            continue

        codes = np.zeros((args.M, 14), dtype=np.int32)
        for i, a in enumerate(actions):
            codes[i, :6] = np.asarray([int(x) for x in a["L"]], dtype=np.int32)
            codes[i, 7:13] = np.asarray([int(x) for x in a["R"]], dtype=np.int32)
            codes[i, 6] = int(a["LG"]) * (args.bins - 1)
            codes[i, 13] = int(a["RG"]) * (args.bins - 1)

        # Hold/repeat: spread M waypoints over the full episode length.
        actions_wp = quant.decode(codes).astype(np.float32)
        actions_full = upsample_waypoints_piecewise_constant(actions_wp, T=args.episode_len)

        BOX_POSE[0] = env0
        res = replay_actions(
            actions_full,
            episode_len=args.episode_len,
            video_path=f"artifacts/replays/eval_icl_json_demo{len(demo_ids)}_test{ep}_idx{idx}.mp4",
        )

        successes += int(res.success)
        rows.append(
            {
                "episode_id": ep,
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
        "demos_dir": args.demos_dir,
        "tests_dir": args.tests_dir,
        "n_demos": len(demo_ids),
        "n_tests": len(test_ids),
        "K_keypoints": args.K,
        "M_actions": args.M,
        "success_rate": successes / max(1, len(test_ids)),
        "parse_error_rate": parse_errors / max(1, len(test_ids)),
        "avg_gen_time_s": float(np.mean(gen_times)) if gen_times else None,
        "out_csv": args.out_csv,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

