"""Print KAT tokens for one episode and (optionally) save a keypoint overlay PNG.

  python3 scripts/show_tokens.py --episode_hdf5 data/.../episode_0.hdf5 \
      --K 10 --M 20 --bins 32 --obs_bins 16 \
      --overlay artifacts/keypoints_overlay.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode
from act_kat.episodes import read_top_frame
from act_kat.icl import anchor_descriptors_from_episode, tokenize_obs
from act_kat.keypoint_anchors import localize_anchors


def main() -> None:
    ap = argparse.ArgumentParser(description="Print OBS + ACT tokens for one episode.")
    ap.add_argument("--episode_hdf5", required=True)
    ap.add_argument("--K", type=int, default=10, help="number of anchored keypoints")
    ap.add_argument("--M", type=int, default=20, help="number of action waypoints")
    ap.add_argument("--bins", type=int, default=32, help="action bins")
    ap.add_argument("--obs_bins", type=int, default=16, help="x/y/d bins")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--overlay", default=None,
        help="optional PNG path; if set, draws keypoint indices on the top image",
    )
    args = ap.parse_args()

    anchor_desc = anchor_descriptors_from_episode(
        args.episode_hdf5, args.K, device=args.device
    )
    with h5py.File(args.episode_hdf5, "r") as root:
        img, depth = read_top_frame(root, t=0)
        actions = root["/action"][()].astype(np.float32)

    obs_tok = tokenize_obs(
        img, anchor_desc=anchor_desc, depth=depth,
        device=args.device, obs_bins=args.obs_bins,
    )
    quant = ActionQuantizer(
        mins=actions.min(axis=0), maxs=actions.max(axis=0), bins=args.bins
    )
    _, act_tok = action_tokens_from_episode(actions, quantizer=quant, M=args.M)

    print(obs_tok)
    print(act_tok)

    if args.overlay:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(args.overlay) or ".", exist_ok=True)
        kps, _ = localize_anchors(img, anchor_desc, device=args.device)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Anchored keypoints (K={args.K})")
        ax.scatter(
            [kp.x_px for kp in kps], [kp.y_px for kp in kps],
            s=60, c="lime", edgecolors="black", linewidths=1.0,
        )
        for i, kp in enumerate(kps):
            ax.text(kp.x_px + 4, kp.y_px - 4, str(i),
                    color="yellow", fontsize=10, weight="bold")
        fig.tight_layout()
        fig.savefig(args.overlay, dpi=200)
        plt.close(fig)
        print(f"Saved overlay: {args.overlay}")


if __name__ == "__main__":
    main()
