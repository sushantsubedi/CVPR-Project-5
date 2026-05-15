from __future__ import annotations

import os
from typing import List, Optional, Tuple

import h5py
import numpy as np


def list_episode_ids(dataset_dir: str) -> List[int]:
    out: List[int] = []
    for name in sorted(os.listdir(dataset_dir)):
        if not (name.startswith("episode_") and name.endswith(".hdf5")):
            continue
        try:
            out.append(int(name[len("episode_") : -len(".hdf5")]))
        except ValueError:
            continue
    return sorted(out)


def episode_path(dataset_dir: str, episode_id: int) -> str:
    return os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")


def read_top_frame(
    root: h5py.File, t: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img = root["/observations/images/top"][t]
    depth = None
    if "/observations/depths/top" in root:
        depth = root["/observations/depths/top"][t]
    return img, depth


def read_env_state0(root: h5py.File) -> np.ndarray:
    return np.array(root.attrs["env_state0"], dtype=np.float32)


def read_qpos0(root: h5py.File, t: int = 0) -> np.ndarray:
    return np.array(root["/observations/qpos"][int(t)], dtype=np.float32)
