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
from act_kat.replay import replay_actions, upsample_waypoints
from sim_env import BOX_POSE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument(
        "--save_video", action="store_true",
        help="save replay video(s) under artifacts/replays/gt_<folder>_ep<id>.mp4",
    )
    ap.add_argument(
        "--num_save", type=int, default=1,
        help="number of episodes to save a video for when --save_video is set",
    )
    ap.add_argument(
        "--upsample", choices=["linear", "hold"], default="linear",
        help="how to reconstruct full-rate actions from stored keyframes when "
             "downsample_rate>1; linear matches the original smooth trajectory closely",
    )
    args = ap.parse_args()

    ep_ids = list_episode_ids(args.dataset_dir)
    tag = os.path.basename(os.path.normpath(args.dataset_dir))
    out_csv = args.out_csv or f"artifacts/results_gt_replay_{tag}.csv"
    n_saved = 0

    rows = []
    for ep in ep_ids:
        path = episode_path(args.dataset_dir, ep)
        with h5py.File(path, "r") as root:
            actions = root["/action"][()].astype("float32")
            env0 = read_env_state0(root)
            # record_sim_episodes downsamples the recorded trajectory by this
            # factor before saving; full-rate replay = stored frames × ds.
            ds = int(root.attrs.get("downsample_rate", 1))
        BOX_POSE[0] = env0

        T_full = int(actions.shape[0]) * max(1, ds)
        T = min(args.episode_len, T_full)
        if ds > 1:
            actions_run = upsample_waypoints(actions, T_full, mode=args.upsample)[:T]
        else:
            actions_run = actions[:T]

        save_video = args.save_video and n_saved < args.num_save
        video_path = f"artifacts/replays/gt_{tag}_ep{ep}.mp4"
        res = replay_actions(
            actions_run, episode_len=T,
            save_video=save_video, video_path=video_path,
        )
        if save_video:
            n_saved += 1
            print(f"  saved video: {video_path}")
        rows.append({
            "episode_id": ep,
            "downsample_rate": ds,
            "success": int(res.success),
            "max_reward": res.max_reward,
            "steps": res.steps,
        })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode_id", "downsample_rate", "success", "max_reward", "steps"],
        )
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
