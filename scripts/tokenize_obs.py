import argparse
import os
import sys

import h5py

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.episodes import read_top_frame
from act_kat.icl import anchor_descriptors_from_episode, tokenize_obs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print DINO anchored keypoint OBS tokens (paper-style).",
    )
    ap.add_argument("--episode_hdf5", type=str, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--obs_bins", type=int, default=64)
    ap.add_argument(
        "--token_format",
        type=str,
        default="keypoints",
        choices=["keypoints", "legacy"],
    )
    args = ap.parse_args()

    anchor_desc = anchor_descriptors_from_episode(
        args.episode_hdf5, args.k, device=args.device
    )
    with h5py.File(args.episode_hdf5, "r") as root:
        img, depth = read_top_frame(root, t=0)

    print(
        tokenize_obs(
            img,
            depth=depth,
            k=args.k,
            device=args.device,
            obs_bins=args.obs_bins,
            anchor_desc=anchor_desc,
            token_format=args.token_format,
        )
    )


if __name__ == "__main__":
    main()
