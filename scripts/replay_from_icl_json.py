import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer
from act_kat.episodes import read_env_state0
from act_kat.icl import replay_from_icl_json_file, replay_from_icl_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode ICL JSON actions and replay in sim.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--icl_json", type=str, help="Path to JSON file")
    g.add_argument("--icl_json_text", type=str, help="Inline JSON string")
    g.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    ap.add_argument("--quantizer_json", type=str, required=True)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument(
        "--upsample",
        type=str,
        default="linear",
        choices=["linear", "hold"],
        help="Waypoint expansion: linear interpolation (default) or hold",
    )
    ap.add_argument("--video_path", type=str, default="artifacts/replays/replay_from_icl_json.mp4")
    ap.add_argument("--box_pose_episode_hdf5", type=str, required=True)
    args = ap.parse_args()

    with open(args.quantizer_json, "r", encoding="utf-8") as f:
        qj = json.load(f)
    quant = ActionQuantizer(
        mins=np.asarray(qj["mins"], dtype=np.float32),
        maxs=np.asarray(qj["maxs"], dtype=np.float32),
        bins=args.bins,
    )

    with h5py.File(args.box_pose_episode_hdf5, "r") as root:
        env0 = read_env_state0(root)

    if args.stdin:
        icl_text = sys.stdin.read()
        kw = dict(
            bins=args.bins,
            episode_len=args.episode_len,
            env_state0=env0,
            video_path=args.video_path,
            upsample=args.upsample,
        )
        res = replay_from_icl_text(icl_text, quant, **kw)
    elif args.icl_json_text:
        res = replay_from_icl_text(
            args.icl_json_text,
            quant,
            bins=args.bins,
            episode_len=args.episode_len,
            env_state0=env0,
            video_path=args.video_path,
            upsample=args.upsample,
        )
    else:
        res = replay_from_icl_json_file(
            args.icl_json,
            quant,
            bins=args.bins,
            episode_len=args.episode_len,
            env_state0=env0,
            video_path=args.video_path,
            upsample=args.upsample,
        )
    print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
