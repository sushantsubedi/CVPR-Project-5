"""Decode-free upper bound: replay stored /action from HDF5 to verify env setup.

Outputs:
  artifacts/results_gt_replay_<folder>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from act_kat.episodes import episode_path, list_episode_ids, read_env_state0
from act_kat.replay import replay_actions
from sim_env import BOX_POSE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    ep_ids = list_episode_ids(args.dataset_dir)
    tag = os.path.basename(os.path.normpath(args.dataset_dir))
    out_csv = args.out_csv or f"artifacts/results_gt_replay_{tag}.csv"

    rows = []
    for ep in ep_ids:
        path = episode_path(args.dataset_dir, ep)
        with h5py.File(path, "r") as root:
            actions = root["/action"][()].astype("float32")
            env0 = read_env_state0(root)
        BOX_POSE[0] = env0
        T = min(args.episode_len, int(actions.shape[0]))
        res = replay_actions(actions[:T], episode_len=T, save_video=False)
        rows.append({
            "episode_id": ep,
            "success": int(res.success),
            "max_reward": res.max_reward,
            "steps": res.steps,
        })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "success", "max_reward", "steps"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    sr = sum(r["success"] for r in rows) / max(1, n)
    print(json.dumps({
        "dataset_dir": args.dataset_dir,
        "out_csv": out_csv,
        "n": n,
        "success_rate": sr,
    }, indent=2))


if __name__ == "__main__":
    main()
