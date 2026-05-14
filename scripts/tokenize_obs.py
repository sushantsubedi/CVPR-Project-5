import argparse
import os
import sys

import h5py

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.vision_tokens import tokenize_keypoints_2d, tokenize_keypoints_2d_with_depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_hdf5", type=str, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--obs_bins", type=int, default=64)
    args = ap.parse_args()

    with h5py.File(args.episode_hdf5, "r") as root:
        img = root["/observations/images/top"][0]
        depth = None
        if "/observations/depths/top" in root:
            depth = root["/observations/depths/top"][0]

    if depth is not None and not (depth.shape[0] == 0):
        _, tok = tokenize_keypoints_2d_with_depth(
            img, depth=depth, k=args.k, device=args.device, obs_bins=args.obs_bins
        )
    else:
        _, tok = tokenize_keypoints_2d(img, k=args.k, device=args.device, obs_bins=args.obs_bins)
    print(tok)


if __name__ == "__main__":
    main()

