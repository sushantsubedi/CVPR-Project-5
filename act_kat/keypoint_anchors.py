"""
Paper-aligned keypoints (KAT / dino-vit-features style).

1. Pick K anchor descriptors from bidirectional correspondences between two views.
2. On every new frame, localize each anchor by cosine nearest-neighbor on the patch grid.

This keeps KP index `i` tied to the same semantic point across demos and queries,
unlike per-frame FPS which reorders points arbitrarily.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

from act_kat.vision_tokens import (
    Keypoint2D,
    _quantize_01,
    extract_patch_descriptors_dinov2,
    farthest_point_sampling_cosine,
    patch_index_to_pixel,
)


def descriptor_grid(
    rgb_uint8: np.ndarray,
    device: torch.device,
    image_size: int = 224,
) -> Tuple[torch.Tensor, int, int]:
    """Returns (Hpatch, Wpatch, D) L2-normalized descriptors."""
    desc_flat, Hpatch, Wpatch = extract_patch_descriptors_dinov2(
        rgb_uint8, device=device, image_size=image_size
    )
    D = desc_flat.shape[1]
    return desc_flat.reshape(Hpatch, Wpatch, D), Hpatch, Wpatch


def _buddy_mask(desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
    """Bidirectional nearest-neighbor (best-buddies) mask on patch tokens."""
    n = desc_a.shape[0]
    sim = desc_a @ desc_b.T
    nn_a_to_b = sim.argmax(dim=1)
    nn_b_to_a = sim.argmax(dim=0)
    idx = torch.arange(n, device=desc_a.device)
    return nn_b_to_a[nn_a_to_b] == idx


def select_anchor_patch_indices(
    img_a: np.ndarray,
    img_b: np.ndarray,
    k: int,
    device: str = "cpu",
    *,
    image_size: int = 224,
) -> np.ndarray:
    """
    K patch indices (into flattened H*W grid of img_a) chosen via best-buddies + FPS.
    Mirrors keypoint_utils.extract_descriptors + correspondence selection.
    """
    dev = torch.device(device)
    desc_a, Ha, Wa = descriptor_grid(img_a, dev, image_size=image_size)
    desc_b, Hb, Wb = descriptor_grid(img_b, dev, image_size=image_size)
    if (Ha, Wa) != (Hb, Wb):
        raise ValueError("image pair must yield the same patch grid after resize")

    flat_a = desc_a.reshape(-1, desc_a.shape[-1])
    flat_b = desc_b.reshape(-1, desc_b.shape[-1])
    buddies = _buddy_mask(flat_a, flat_b)
    buddy_idx = torch.nonzero(buddies, as_tuple=False).squeeze(1)
    if buddy_idx.numel() == 0:
        buddy_idx = torch.arange(flat_a.shape[0], device=dev)

    k_eff = min(k, int(buddy_idx.numel()))
    sub = flat_a.index_select(0, buddy_idx)
    sel_local = farthest_point_sampling_cosine(sub, k=k_eff, start_idx=0)
    anchors = buddy_idx.index_select(0, sel_local)
    return anchors.detach().cpu().numpy().astype(np.int64)


def anchor_descriptors_from_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    k: int,
    device: str = "cpu",
    *,
    image_size: int = 224,
) -> np.ndarray:
    """(K, D) float32 anchor descriptor vectors (reference frame = img_a)."""
    dev = torch.device(device)
    desc_a, Ha, Wa = descriptor_grid(img_a, dev, image_size=image_size)
    patch_idx = select_anchor_patch_indices(img_a, img_b, k, device=device, image_size=image_size)
    flat_a = desc_a.reshape(-1, desc_a.shape[-1])
    anchors = flat_a[torch.as_tensor(patch_idx, device=dev)].detach().cpu().numpy().astype(np.float32)
    return anchors


def anchor_descriptors_from_episode(
    episode_hdf5: str,
    k: int,
    device: str = "cpu",
    *,
    frame_a: int = 0,
    frame_b: Optional[int] = None,
    image_size: int = 224,
) -> np.ndarray:
    """Build anchors from two frames in one episode (default t=0 and t=mid)."""
    with h5py.File(episode_hdf5, "r") as root:
        imgs = root["/observations/images/top"]
        if frame_b is None:
            frame_b = max(0, int(imgs.shape[0]) // 2)
        img_a = imgs[int(frame_a)]
        img_b = imgs[int(frame_b)]
    return anchor_descriptors_from_pair(img_a, img_b, k, device=device, image_size=image_size)


def localize_anchors(
    rgb_uint8: np.ndarray,
    anchor_desc: np.ndarray,
    device: str = "cpu",
    *,
    image_size: int = 224,
) -> Tuple[List[Keypoint2D], np.ndarray]:
    """
    For each anchor descriptor, find best-matching patch on this image (cosine NN).
    Returns keypoints and patch indices (K,).
    """
    dev = torch.device(device)
    desc_grid, Hpatch, Wpatch = descriptor_grid(rgb_uint8, dev, image_size=image_size)
    flat = desc_grid.reshape(-1, desc_grid.shape[-1])
    anchors = torch.as_tensor(anchor_desc, dtype=torch.float32, device=dev)
    anchors = torch.nn.functional.normalize(anchors, dim=-1)

    patch_idx = []
    H, W, _ = rgb_uint8.shape
    kps: List[Keypoint2D] = []
    for i in range(anchors.shape[0]):
        sim = flat @ anchors[i]
        pi = int(sim.argmax().item())
        patch_idx.append(pi)
        x_px, y_px = patch_index_to_pixel(pi, Hpatch, Wpatch, H, W)
        kps.append(
            Keypoint2D(
                x_px=x_px,
                y_px=y_px,
                x_norm=float(x_px) / float(max(1, W - 1)),
                y_norm=float(y_px) / float(max(1, H - 1)),
            )
        )
    return kps, np.asarray(patch_idx, dtype=np.int64)


def tokenize_anchored_keypoints(
    rgb_uint8: np.ndarray,
    anchor_desc: np.ndarray,
    *,
    depth: Optional[np.ndarray] = None,
    device: str = "cpu",
    obs_bins: int = 64,
    depth_clip: Tuple[float, float] = (0.0, 2.0),
    image_size: int = 224,
) -> Tuple[List[Keypoint2D], str]:
    kps, _ = localize_anchors(rgb_uint8, anchor_desc, device=device, image_size=image_size)

    lines = ["OBS_START"]
    if depth is not None and np.isfinite(np.asarray(depth)).any():
        d0, d1 = depth_clip
        depth_clipped = np.clip(depth.astype(np.float32), d0, d1)
        d_norm = (depth_clipped - d0) / max(1e-6, (d1 - d0))
        for i, kp in enumerate(kps):
            dn = float(d_norm[kp.y_px, kp.x_px])
            lines.append(
                f"KP {i} x={_quantize_01(kp.x_norm, obs_bins)} "
                f"y={_quantize_01(kp.y_norm, obs_bins)} "
                f"d={_quantize_01(dn, obs_bins)}"
            )
    else:
        for i, kp in enumerate(kps):
            lines.append(
                f"KP {i} x={_quantize_01(kp.x_norm, obs_bins)} y={_quantize_01(kp.y_norm, obs_bins)}"
            )
    lines.append("OBS_END")
    return kps, "\n".join(lines) + "\n"
