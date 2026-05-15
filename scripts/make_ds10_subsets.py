import argparse
import os
import shutil
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from act_kat.episodes import list_episode_ids


def _clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _create_subset(base_dir: str, out_dir: str, ids: List[int]) -> None:
    _clear_dir(out_dir)
    wanted = set(ids)
    for ep_id in list_episode_ids(base_dir):
        if ep_id not in wanted:
            continue
        src_path = os.path.join(base_dir, f"episode_{ep_id}.hdf5")
        dst_path = os.path.join(out_dir, f"episode_{ep_id}.hdf5")
        _link_or_copy(src_path, dst_path)

    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        for ep_id in ids:
            f.write(f"{ep_id}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument(
        "--out_root",
        type=str,
        default="data/transfer_cube/ds10/subsets",
    )
    args = ap.parse_args()

    base_eps = list_episode_ids(args.base_dir)
    if len(base_eps) < 25:
        raise SystemExit(f"Need >=25 episodes in base_dir, found {len(base_eps)}")

    os.makedirs(args.out_root, exist_ok=True)
    _create_subset(args.base_dir, os.path.join(args.out_root, "demos_5"), base_eps[:5])
    _create_subset(args.base_dir, os.path.join(args.out_root, "demos_10"), base_eps[:10])
    _create_subset(args.base_dir, os.path.join(args.out_root, "demos_20"), base_eps[:20])
    _create_subset(args.base_dir, os.path.join(args.out_root, "test_5"), base_eps[20:25])
    print("Created subsets under:", args.out_root)


if __name__ == "__main__":
    main()
