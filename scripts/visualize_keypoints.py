import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.vision_tokens import tokenize_keypoints_2d, tokenize_keypoints_2d_with_depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_hdf5", type=str, required=True)
    ap.add_argument("--t", type=int, default=0, help="Timestep index to visualize")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--obs_bins", type=int, default=64)
    ap.add_argument("--out_png", type=str, default="artifacts/keypoints_overlay.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)

    with h5py.File(args.episode_hdf5, "r") as root:
        img = root["/observations/images/top"][args.t]
        depth = root["/observations/depths/top"][args.t] if "/observations/depths/top" in root else None

    if depth is not None and np.isfinite(np.asarray(depth)).any():
        kps, tok = tokenize_keypoints_2d_with_depth(
            img, depth=depth, k=args.k, device=args.device, obs_bins=args.obs_bins
        )
    else:
        kps, tok = tokenize_keypoints_2d(img, k=args.k, device=args.device, obs_bins=args.obs_bins)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img)
    ax.set_title(f"Keypoints overlay (K={args.k}, t={args.t})")
    ax.axis("off")

    xs = [kp.x_px for kp in kps]
    ys = [kp.y_px for kp in kps]
    ax.scatter(xs, ys, s=60, c="lime", edgecolors="black", linewidths=1.0)
    for i, kp in enumerate(kps):
        ax.text(kp.x_px + 4, kp.y_px - 4, str(i), color="yellow", fontsize=10, weight="bold")

    plt.tight_layout()
    fig.savefig(args.out_png, dpi=200)
    plt.close(fig)

    print(f"Saved overlay: {args.out_png}")
    print("Tokens:")
    print(tok)


if __name__ == "__main__":
    main()

