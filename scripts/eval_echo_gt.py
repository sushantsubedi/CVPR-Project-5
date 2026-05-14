import argparse
import csv
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode, parse_action_block
from act_kat.ollama_client import ollama_chat
from act_kat.prompting import SYSTEM_INSTRUCTION
from act_kat.replay import replay_actions, upsample_waypoints_linear
from sim_env import BOX_POSE


def extract_act_block_from_text(text: str) -> str:
    import re

    pattern = re.compile(r"(?ms)^[ \t]*ACT_START[ \t]*\n.*?^[ \t]*ACT_END[ \t]*$")
    m = pattern.search(text)
    return m.group(0) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemma4:26b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--K", type=int, default=10)  # unused in echo_gt, kept for results schema
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--out_csv", type=str, default="artifacts/results.csv")
    ap.add_argument("--n_eval", type=int, default=5)
    args = ap.parse_args()

    # Build a single quantizer across the dataset (mins/maxs)
    actions_all = []
    for i in range(args.n_eval):
        with h5py.File(os.path.join(args.dataset_dir, f"episode_{i}.hdf5"), "r") as root:
            actions_all.append(root["/action"][()].astype(np.float32))
    actions_all = np.concatenate(actions_all, axis=0)
    quant = ActionQuantizer(
        mins=actions_all.min(axis=0), maxs=actions_all.max(axis=0), bins=args.bins
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    successes = 0
    parse_errors = 0
    gen_times = []

    for ep_id in range(args.n_eval):
        ep_path = os.path.join(args.dataset_dir, f"episode_{ep_id}.hdf5")
        with h5py.File(ep_path, "r") as root:
            act = root["/action"][()].astype(np.float32)
            env0 = np.array(root.attrs["env_state0"], dtype=np.float32)

        # Ground-truth token block (what we ask the LLM to repeat)
        _, act_tok_gt = action_tokens_from_episode(act, quantizer=quant, M=args.M)
        system_msg = (
            "Output EXACTLY the provided ACT block, from ACT_START to ACT_END. No other text."
        )
        user_msg = "Repeat this ACT block exactly:\n\n" + act_tok_gt

        t0 = time.time()
        resp = ollama_chat(
            system=system_msg,
            user=user_msg,
            model=args.model,
            url=args.ollama_url,
            options={"temperature": 0.0, "num_predict": 2048},
            run_id=f"eval_echo_gt_ep{ep_id}",
            artifacts_dir="artifacts",
            timeout_s=600,
        )
        gen_times.append(time.time() - t0)

        msg = resp.get("message", {}) if isinstance(resp, dict) else {}
        text = (msg.get("content") or "") + "\n" + (msg.get("thinking") or "")
        block = extract_act_block_from_text(text)
        codes, errs = parse_action_block(block, M_expected=args.M) if block else (None, ["no_block"])
        if codes is None:
            parse_errors += 1
            rows.append(
                {
                    "episode_id": ep_id,
                    "success": 0,
                    "max_reward": 0.0,
                    "parse_ok": 0,
                    "parse_errors": json.dumps(errs),
                }
            )
            continue

        # map gripper bits -> bin extremes
        codes = codes.copy()
        codes[:, 6] = codes[:, 6] * (args.bins - 1)
        codes[:, 13] = codes[:, 13] * (args.bins - 1)

        actions_wp = quant.decode(codes).astype(np.float32)
        actions = upsample_waypoints_linear(actions_wp, T=args.episode_len)

        BOX_POSE[0] = env0
        res = replay_actions(
            actions,
            episode_len=args.episode_len,
            video_path=f"artifacts/replays/eval_echo_gt_ep{ep_id}.mp4",
        )

        successes += int(res.success)
        rows.append(
            {
                "episode_id": ep_id,
                "success": int(res.success),
                "max_reward": res.max_reward,
                "parse_ok": 1,
                "parse_errors": json.dumps(errs),
            }
        )

    # Write per-episode rows
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode_id", "success", "max_reward", "parse_ok", "parse_errors"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print summary (also useful in logs)
    summary = {
        "dataset_dir": args.dataset_dir,
        "N_demos": "echo_gt",
        "K_keypoints": args.K,
        "M_actions": args.M,
        "downsample_rate": 10,
        "success_rate": successes / max(1, args.n_eval),
        "parse_error_rate": parse_errors / max(1, args.n_eval),
        "avg_gen_time_s": float(np.mean(gen_times)) if gen_times else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

