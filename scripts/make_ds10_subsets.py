import argparse
import os
import shutil
from typing import List, Tuple


def _list_episode_files(dataset_dir: str) -> List[Tuple[int, str]]:
    eps = []
    for name in os.listdir(dataset_dir):
        if not (name.startswith("episode_") and name.endswith(".hdf5")):
            continue
        try:
            ep_id = int(name[len("episode_") : -len(".hdf5")])
        except Exception:
            continue
        eps.append((ep_id, os.path.join(dataset_dir, name)))
    return sorted(eps, key=lambda x: x[0])


def _clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str) -> None:
    # Prefer hardlink to avoid duplication; fall back to copy if unsupported.
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _create_subset(base_dir: str, out_dir: str, ids: List[int]) -> None:
    _clear_dir(out_dir)
    wanted = set(ids)
    for ep_id, src_path in _list_episode_files(base_dir):
        if ep_id not in wanted:
            continue
        dst_path = os.path.join(out_dir, f"episode_{ep_id}.hdf5")
        _link_or_copy(src_path, dst_path)

    # Save manifest for transparency
    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        for ep_id in ids:
            f.write(f"{ep_id}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Directory containing episode_*.hdf5")
    ap.add_argument(
        "--out_root",
        type=str,
        default="data/transfer_cube/ds10/subsets",
        help="Where to create subsets",
    )
    args = ap.parse_args()

    base_eps = [ep_id for ep_id, _ in _list_episode_files(args.base_dir)]
    if len(base_eps) < 25:
        raise SystemExit(f"Need >=25 episodes in base_dir, found {len(base_eps)}")

    os.makedirs(args.out_root, exist_ok=True)

    demos_5 = base_eps[:5]
    demos_10 = base_eps[:10]
    demos_20 = base_eps[:20]
    test_5 = base_eps[20:25]

    _create_subset(args.base_dir, os.path.join(args.out_root, "demos_5"), demos_5)
    _create_subset(args.base_dir, os.path.join(args.out_root, "demos_10"), demos_10)
    _create_subset(args.base_dir, os.path.join(args.out_root, "demos_20"), demos_20)
    _create_subset(args.base_dir, os.path.join(args.out_root, "test_5"), test_5)

    print("Created subsets under:", args.out_root)


if __name__ == "__main__":
    main()

