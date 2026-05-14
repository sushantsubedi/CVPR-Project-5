import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_hdf5", type=str, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--bins", type=int, default=64)
    args = ap.parse_args()

    with h5py.File(args.episode_hdf5, "r") as root:
        a = root["/action"][()].astype(np.float32)

    mins = a.min(axis=0)
    maxs = a.max(axis=0)
    q = ActionQuantizer(mins=mins, maxs=maxs, bins=args.bins)
    _, tok = action_tokens_from_episode(a, quantizer=q, M=args.M)
    print(tok)


if __name__ == "__main__":
    main()

