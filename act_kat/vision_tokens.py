"""DINO ViT-S/16 patch descriptors and shared keypoint primitives.

This module exposes only what `keypoint_anchors` needs:
  - load the local DINO ViT-S/16 checkpoint
  - extract L2-normalized patch descriptors
  - farthest-point-sampling (cosine)
  - patch index -> pixel coords
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import timm
import torch
import torchvision.transforms as T

_REPO_ROOT = Path(__file__).resolve().parents[1]
DINO_CHECKPOINT_PATH = _REPO_ROOT / "assets" / "weights" / "vit_small_patch16_224.dino.pth"
DINO_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
)
DINO_MODEL_NAME = "vit_small_patch16_224.dino"

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


@dataclass(frozen=True)
class Keypoint2D:
    x_px: int
    y_px: int
    x_norm: float
    y_norm: float


def _quantize_01(x: float, bins: int) -> int:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    xf = max(0.0, min(1.0, float(x)))
    return max(0, min(bins - 1, int(round(xf * (bins - 1)))))


def _make_preprocess(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def dino_checkpoint_path() -> Path:
    override = os.environ.get("ACT_KAT_DINO_CHECKPOINT")
    return Path(override) if override else DINO_CHECKPOINT_PATH


def _load_dino_vit_small_patch16(device: torch.device) -> torch.nn.Module:
    cache_key = str(device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    ckpt = dino_checkpoint_path()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"DINO ViT weights not found at {ckpt}.\n"
            "Download once (Meta CDN, ~83MB):\n"
            "  python3 scripts/download_dino_weights.py"
        )
    model = timm.create_model(DINO_MODEL_NAME, pretrained=False, num_classes=0)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    _MODEL_CACHE[cache_key] = model
    return model


@torch.inference_mode()
def extract_patch_descriptors_dinov2(
    rgb_uint8: np.ndarray,
    device: torch.device,
    image_size: int = 224,
) -> Tuple[torch.Tensor, int, int]:
    """Returns (Hpatch*Wpatch, D) L2-normalized descriptors and grid size."""
    if rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must be uint8 HxWx3")

    img = _make_preprocess(image_size=image_size)(rgb_uint8).unsqueeze(0).to(device)
    model = _load_dino_vit_small_patch16(device)
    feats = model.forward_features(img)
    if feats.ndim != 3:
        raise ValueError(f"unexpected forward_features shape: {tuple(feats.shape)}")
    patch = feats[:, 1:, :].squeeze(0) if feats.shape[1] > 1 else feats.squeeze(0)
    n = int(patch.shape[0])
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"unexpected patch token count {n} (not square)")
    patch = torch.nn.functional.normalize(patch.float(), dim=-1)
    return patch, side, side


def farthest_point_sampling_cosine(
    desc: torch.Tensor, k: int, *, start_idx: int = 0
) -> torch.Tensor:
    """desc: (N,D) L2-normalized -> indices (k,)."""
    if desc.ndim != 2:
        raise ValueError("desc must be (N,D)")
    N = desc.shape[0]
    if not (1 <= k <= N):
        raise ValueError(f"k must be in [1,{N}]")
    if not (0 <= start_idx < N):
        raise ValueError("start_idx out of range")

    selected = torch.empty((k,), dtype=torch.long, device=desc.device)
    selected[0] = int(start_idx)
    min_dist = 1.0 - (desc @ desc[selected[0]].unsqueeze(-1)).squeeze(-1)
    for i in range(1, k):
        idx = torch.argmax(min_dist)
        selected[i] = idx
        dist = 1.0 - (desc @ desc[idx].unsqueeze(-1)).squeeze(-1)
        min_dist = torch.minimum(min_dist, dist)
    return selected


def patch_index_to_pixel(
    patch_idx: int, Hpatch: int, Wpatch: int, orig_h: int, orig_w: int
) -> Tuple[int, int]:
    py, px = patch_idx // Wpatch, patch_idx % Wpatch
    x = max(0, min(orig_w - 1, int(round((px + 0.5) * orig_w / Wpatch))))
    y = max(0, min(orig_h - 1, int(round((py + 0.5) * orig_h / Hpatch))))
    return x, y
