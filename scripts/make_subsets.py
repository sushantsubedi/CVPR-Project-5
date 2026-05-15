"""Split a base HDF5 episode folder into demo / test subsets for P5 evaluation.

Default layout (matches `data/transfer_cube/DATA_MANIFEST.md`):
  base/         episode_0 .. episode_24
  subsets/demos_5     episodes 0..4
  subsets/demos_10    episodes 0..9
  subsets/demos_20    episodes 0..19
  subsets/test_5      episodes 20..24   (held out -> "unseen")
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from act_kat.episodes import list_episode_ids


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _make_subset(base_dir: str, out_dir: str, ids: List[int]) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(ids)
    for ep in list_episode_ids(base_dir):
        if ep not in wanted:
            continue
        _link_or_copy(
            os.path.join(base_dir, f"episode_{ep}.hdf5"),
            os.path.join(out_dir, f"episode_{ep}.hdf5"),
        )
    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        for ep in ids:
            f.write(f"{ep}\n")
    print(f"  {out_dir}: {len(ids)} episodes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="folder with episode_*.hdf5")
    ap.add_argument(
        "--out_root", default="data/transfer_cube/subsets",
        help="output root for demos_5/10/20 and test_5",
    )
    args = ap.parse_args()

    ids = list_episode_ids(args.base_dir)
    if len(ids) < 25:
        raise SystemExit(f"Need >=25 episodes in {args.base_dir}, found {len(ids)}")

    os.makedirs(args.out_root, exist_ok=True)
    print(f"Creating subsets under {args.out_root}:")
    _make_subset(args.base_dir, os.path.join(args.out_root, "demos_5"), ids[:5])
    _make_subset(args.base_dir, os.path.join(args.out_root, "demos_10"), ids[:10])
    _make_subset(args.base_dir, os.path.join(args.out_root, "demos_20"), ids[:20])
    _make_subset(args.base_dir, os.path.join(args.out_root, "test_5"), ids[20:25])


if __name__ == "__main__":
    main()
