"""Decode an ICL JSON response and replay it in MuJoCo.

  python3 scripts/replay_response.py \
      --response artifacts/responses/<run_id>.json \
      --demos_dir data/transfer_cube/subsets/demos_5 \
      --test_episode_hdf5 data/transfer_cube/subsets/test_5/episode_20.hdf5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from act_kat.action_tokens import ActionQuantizer
from act_kat.episodes import episode_path, list_episode_ids, read_env_state0
from act_kat.icl import fit_quantizer_from_episodes, replay_response


def _load_quantizer(args: argparse.Namespace) -> ActionQuantizer:
    if args.quantizer_json:
        import numpy as np

        with open(args.quantizer_json, "r", encoding="utf-8") as f:
            qj = json.load(f)
        return ActionQuantizer(
            mins=np.asarray(qj["mins"], dtype=np.float32),
            maxs=np.asarray(qj["maxs"], dtype=np.float32),
            bins=args.bins,
        )
    if args.demos_dir:
        ids = list_episode_ids(args.demos_dir)
        if not ids:
            raise SystemExit(f"No episodes in {args.demos_dir}")
        return fit_quantizer_from_episodes(
            [episode_path(args.demos_dir, ep) for ep in ids], bins=args.bins
        )
    raise SystemExit("provide --demos_dir or --quantizer_json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay an ICL JSON response in sim.")
    ap.add_argument("--response", "--llm_response", dest="response", required=True,
                    help="path to artifacts/responses/<run_id>.json (or '-' for stdin)")
    ap.add_argument("--test_episode_hdf5", required=True,
                    help="HDF5 used to seed initial box pose (BOX_POSE)")
    ap.add_argument("--demos_dir", default=None,
                    help="folder of demo HDF5s; quantizer is fit from these")
    ap.add_argument("--quantizer_json", default=None,
                    help="precomputed quantizer (mins/maxs); alt to --demos_dir")
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--M", type=int, default=None,
                    help="if set, strictly require M waypoints in the response")
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--upsample", choices=["linear", "hold"], default="linear")
    ap.add_argument("--video_path", default=None)
    args = ap.parse_args()

    quant = _load_quantizer(args)

    with h5py.File(args.test_episode_hdf5, "r") as root:
        env0 = read_env_state0(root)

    text = sys.stdin.read() if args.response == "-" else open(args.response).read()

    video_path = args.video_path or os.path.join(
        "artifacts/replays",
        os.path.splitext(os.path.basename(args.response))[0] + "_replay.mp4",
    )
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    res = replay_response(
        text, quant,
        env_state0=env0, video_path=video_path,
        bins=args.bins, M=args.M, episode_len=args.episode_len, upsample=args.upsample,
    )
    print(json.dumps({
        "video_path": res.video_path,
        "success": res.success,
        "max_reward": res.max_reward,
        "steps": res.steps,
    }, indent=2))


if __name__ == "__main__":
    main()
