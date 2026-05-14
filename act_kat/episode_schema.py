from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "/observations/qpos",
    "/observations/qvel",
    "/action",
)


@dataclass
class EpisodeSchemaReport:
    dataset_path: str
    ok: bool
    errors: List[str]
    created_at_utc: str

    # attrs
    attr_sim: Optional[bool]
    attr_downsample_rate: Optional[int]

    # camera keys
    cameras: List[str]

    # shapes/dtypes
    T: Optional[int]
    qpos_shape: Optional[Tuple[int, ...]]
    qpos_dtype: Optional[str]
    qvel_shape: Optional[Tuple[int, ...]]
    qvel_dtype: Optional[str]
    action_shape: Optional[Tuple[int, ...]]
    action_dtype: Optional[str]
    image_shapes: Dict[str, Tuple[int, ...]]
    image_dtypes: Dict[str, str]

    # basic stats / sanity
    has_nans: Optional[bool]
    qpos_min: Optional[float]
    qpos_max: Optional[float]
    action_min: Optional[float]
    action_max: Optional[float]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            y = float(x)
            if math.isfinite(y):
                return y
            return None
        return float(x)
    except Exception:
        return None


def inspect_episode_hdf5(dataset_path: str) -> EpisodeSchemaReport:
    errors: List[str] = []
    cameras: List[str] = []
    image_shapes: Dict[str, Tuple[int, ...]] = {}
    image_dtypes: Dict[str, str] = {}

    attr_sim: Optional[bool] = None
    attr_downsample_rate: Optional[int] = None

    T: Optional[int] = None
    qpos_shape: Optional[Tuple[int, ...]] = None
    qvel_shape: Optional[Tuple[int, ...]] = None
    action_shape: Optional[Tuple[int, ...]] = None
    qpos_dtype: Optional[str] = None
    qvel_dtype: Optional[str] = None
    action_dtype: Optional[str] = None

    has_nans: Optional[bool] = None
    qpos_min: Optional[float] = None
    qpos_max: Optional[float] = None
    action_min: Optional[float] = None
    action_max: Optional[float] = None

    try:
        with h5py.File(dataset_path, "r") as root:
            # attrs
            if "sim" in root.attrs:
                try:
                    attr_sim = bool(root.attrs["sim"])
                except Exception:
                    errors.append("attr_sim_unreadable")
            else:
                errors.append("missing_attr_sim")

            if "downsample_rate" in root.attrs:
                try:
                    attr_downsample_rate = int(root.attrs["downsample_rate"])
                except Exception:
                    errors.append("attr_downsample_rate_unreadable")
            else:
                errors.append("missing_attr_downsample_rate")

            # required datasets
            for ds in REQUIRED_DATASETS:
                if ds not in root:
                    errors.append(f"missing_dataset:{ds}")

            if "/observations/images" not in root:
                errors.append("missing_group:/observations/images")
            else:
                cameras = sorted(list(root["/observations/images"].keys()))
                if len(cameras) == 0:
                    errors.append("no_cameras_found")

            # read metadata
            if "/observations/qpos" in root:
                qpos = root["/observations/qpos"]
                qpos_shape = tuple(qpos.shape)
                qpos_dtype = str(qpos.dtype)
                if len(qpos_shape) >= 1:
                    T = int(qpos_shape[0])

            if "/observations/qvel" in root:
                qvel = root["/observations/qvel"]
                qvel_shape = tuple(qvel.shape)
                qvel_dtype = str(qvel.dtype)

            if "/action" in root:
                action = root["/action"]
                action_shape = tuple(action.shape)
                action_dtype = str(action.dtype)

            # image metadata
            if "/observations/images" in root:
                for cam in root["/observations/images"].keys():
                    ds = root[f"/observations/images/{cam}"]
                    image_shapes[cam] = tuple(ds.shape)
                    image_dtypes[cam] = str(ds.dtype)

            # consistency checks
            if T is not None:
                if qvel_shape is not None and len(qvel_shape) >= 1 and int(qvel_shape[0]) != T:
                    errors.append("inconsistent_T:qvel")
                if action_shape is not None and len(action_shape) >= 1 and int(action_shape[0]) != T:
                    errors.append("inconsistent_T:action")
                for cam, shp in image_shapes.items():
                    if len(shp) >= 1 and int(shp[0]) != T:
                        errors.append(f"inconsistent_T:image:{cam}")

            # NaNs + ranges (only qpos/action; keep it light)
            try:
                has_nans = False
                if "/observations/qpos" in root:
                    qpos_arr = root["/observations/qpos"][()]
                    has_nans = has_nans or bool(np.isnan(qpos_arr).any())
                    qpos_min = _safe_float(np.min(qpos_arr))
                    qpos_max = _safe_float(np.max(qpos_arr))
                if "/action" in root:
                    act_arr = root["/action"][()]
                    has_nans = has_nans or bool(np.isnan(act_arr).any())
                    action_min = _safe_float(np.min(act_arr))
                    action_max = _safe_float(np.max(act_arr))
            except Exception:
                errors.append("stats_failed")

    except FileNotFoundError:
        errors.append("file_not_found")
    except OSError as e:
        errors.append(f"oserror:{type(e).__name__}")
    except Exception as e:
        errors.append(f"unexpected:{type(e).__name__}")

    ok = len(errors) == 0
    return EpisodeSchemaReport(
        dataset_path=os.path.abspath(dataset_path),
        ok=ok,
        errors=errors,
        created_at_utc=_utc_now_iso(),
        attr_sim=attr_sim,
        attr_downsample_rate=attr_downsample_rate,
        cameras=cameras,
        T=T,
        qpos_shape=qpos_shape,
        qpos_dtype=qpos_dtype,
        qvel_shape=qvel_shape,
        qvel_dtype=qvel_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        image_shapes=image_shapes,
        image_dtypes=image_dtypes,
        has_nans=has_nans,
        qpos_min=qpos_min,
        qpos_max=qpos_max,
        action_min=action_min,
        action_max=action_max,
    )


def _iter_episode_paths(dataset_dir: str) -> List[str]:
    # Expect files like episode_0.hdf5
    out: List[str] = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.startswith("episode_") and name.endswith(".hdf5"):
            out.append(os.path.join(dataset_dir, name))
    return out


def write_schema_report(dataset_dir: str, out_jsonl: str) -> None:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    episode_paths = _iter_episode_paths(dataset_dir)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in episode_paths:
            rep = inspect_episode_hdf5(p)
            f.write(json.dumps(asdict(rep)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/schema_report.jsonl",
        help="Output JSONL path (default: artifacts/schema_report.jsonl)",
    )
    args = ap.parse_args()
    write_schema_report(args.dataset_dir, args.out)


if __name__ == "__main__":
    main()

