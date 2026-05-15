from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import timm
import torchvision.transforms as T

_REPO_ROOT = Path(__file__).resolve().parents[1]
DINO_CHECKPOINT_PATH = _REPO_ROOT / "assets" / "weights" / "vit_small_patch16_224.dino.pth"
# Official DINO checkpoint (Meta CDN) — same weights timm uses for vit_small_patch16_224.dino
DINO_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
)
DINO_MODEL_NAME = "vit_small_patch16_224.dino"

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


@dataclass(frozen=True)
class Keypoint2D:
    # Pixel coordinates in original image
    x_px: int
    y_px: int

    # Normalized to [0, 1]
    x_norm: float
    y_norm: float


@dataclass(frozen=True)
class Keypoint2DDepth(Keypoint2D):
    # Depth value at keypoint (raw simulator units / meters depending on renderer)
    depth: float


def _quantize_01(x: float, bins: int) -> int:
    """
    Quantize a scalar in [0,1] to an integer in [0, bins-1].
    Clamps out-of-range inputs.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    xf = float(x)
    xf = 0.0 if xf < 0.0 else (1.0 if xf > 1.0 else xf)
    q = int(round(xf * (bins - 1)))
    return max(0, min(bins - 1, q))


def _make_preprocess(image_size: int = 224) -> T.Compose:
    # DINO-style normalization (ImageNet)
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
    """Load ViT-S/16 DINO from local weights only (no Hugging Face / network at inference)."""
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
    model.eval()
    model.to(device)
    _MODEL_CACHE[cache_key] = model
    return model


@torch.inference_mode()
def extract_patch_descriptors_dinov2(
    rgb_uint8: np.ndarray,
    device: torch.device,
    image_size: int = 224,
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
      desc: (Hpatch*Wpatch, D) float32, L2-normalized
      Hpatch, Wpatch: patch grid size
    """
    if rgb_uint8.dtype != np.uint8:
        raise ValueError("rgb_uint8 must be uint8 HxWx3")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must be HxWx3")

    preprocess = _make_preprocess(image_size=image_size)
    img = preprocess(rgb_uint8).unsqueeze(0).to(device)  # (1,3,H,W)

    model = _load_dino_vit_small_patch16(device)
    feats = model.forward_features(img)  # (1, N+1, D) or (1, N, D) depending on timm version
    if feats.ndim != 3:
        raise ValueError(f"Unexpected forward_features shape: {tuple(feats.shape)}")

    # timm ViT forward_features typically includes CLS token at index 0
    if feats.shape[1] > 1:
        patch = feats[:, 1:, :].squeeze(0)  # (N, D)
    else:
        patch = feats.squeeze(0)

    n = int(patch.shape[0])
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"Unexpected patch token count {n} (not square)")
    Hpatch = Wpatch = side

    patch = patch.float()
    patch = torch.nn.functional.normalize(patch, dim=-1)
    return patch, Hpatch, Wpatch


def farthest_point_sampling_cosine(desc: torch.Tensor, k: int, *, start_idx: int = 0) -> torch.Tensor:
    """
    desc: (N, D) L2-normalized
    returns indices: (k,)
    """
    if desc.ndim != 2:
        raise ValueError("desc must be (N,D)")
    N = desc.shape[0]
    if not (1 <= k <= N):
        raise ValueError(f"k must be in [1,{N}]")

    # cosine distance = 1 - cos_sim
    # initialize from a caller-chosen index (important when desc is a filtered candidate set)
    selected = torch.empty((k,), dtype=torch.long, device=desc.device)
    if not (0 <= start_idx < N):
        raise ValueError("start_idx out of range")
    selected[0] = int(start_idx)

    # min distance to selected set for each point
    cos_sim = desc @ desc[selected[0]].unsqueeze(-1)  # (N,1)
    min_dist = 1.0 - cos_sim.squeeze(-1)  # (N,)

    for i in range(1, k):
        idx = torch.argmax(min_dist)
        selected[i] = idx
        cos_sim = desc @ desc[idx].unsqueeze(-1)
        dist = 1.0 - cos_sim.squeeze(-1)
        min_dist = torch.minimum(min_dist, dist)

    return selected


def patch_index_to_pixel(
    patch_idx: int,
    Hpatch: int,
    Wpatch: int,
    orig_h: int,
    orig_w: int,
) -> Tuple[int, int]:
    py = patch_idx // Wpatch
    px = patch_idx % Wpatch

    # Map patch center to original image coordinates.
    # Using proportional mapping (not exact intrinsics).
    x = int(round((px + 0.5) * orig_w / Wpatch))
    y = int(round((py + 0.5) * orig_h / Hpatch))
    x = max(0, min(orig_w - 1, x))
    y = max(0, min(orig_h - 1, y))
    return x, y


def _downsample_to_patch_grid_mean(x: np.ndarray, Hpatch: int, Wpatch: int) -> np.ndarray:
    """
    Downsample an HxW array to (Hpatch, Wpatch) by simple block mean.
    Assumes Hpatch|H and Wpatch|W approximately; we crop to fit.
    """
    if x.ndim != 2:
        raise ValueError("x must be HxW")
    H, W = x.shape
    ph = H // Hpatch
    pw = W // Wpatch
    if ph <= 0 or pw <= 0:
        raise ValueError("patch grid larger than input")
    Hc = ph * Hpatch
    Wc = pw * Wpatch
    xc = x[:Hc, :Wc]
    return xc.reshape(Hpatch, ph, Wpatch, pw).mean(axis=(1, 3))


def _edge_strength_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Returns per-pixel edge magnitude in [0,1] (roughly), using Sobel on grayscale.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("rgb_uint8 must be HxWx3")
    g = (
        0.2989 * rgb_uint8[:, :, 0].astype(np.float32)
        + 0.5870 * rgb_uint8[:, :, 1].astype(np.float32)
        + 0.1140 * rgb_uint8[:, :, 2].astype(np.float32)
    )
    # Sobel kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # lightweight conv (no scipy): pad + manual convolution
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    gx = (
        kx[0, 0] * gp[:-2, :-2]
        + kx[0, 1] * gp[:-2, 1:-1]
        + kx[0, 2] * gp[:-2, 2:]
        + kx[1, 0] * gp[1:-1, :-2]
        + kx[1, 1] * gp[1:-1, 1:-1]
        + kx[1, 2] * gp[1:-1, 2:]
        + kx[2, 0] * gp[2:, :-2]
        + kx[2, 1] * gp[2:, 1:-1]
        + kx[2, 2] * gp[2:, 2:]
    )
    gy = (
        ky[0, 0] * gp[:-2, :-2]
        + ky[0, 1] * gp[:-2, 1:-1]
        + ky[0, 2] * gp[:-2, 2:]
        + ky[1, 0] * gp[1:-1, :-2]
        + ky[1, 1] * gp[1:-1, 1:-1]
        + ky[1, 2] * gp[1:-1, 2:]
        + ky[2, 0] * gp[2:, :-2]
        + ky[2, 1] * gp[2:, 1:-1]
        + ky[2, 2] * gp[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    mag /= float(np.percentile(mag, 99) + 1e-6)
    return np.clip(mag, 0.0, 1.0)


def _choose_candidate_patch_indices(
    *,
    Hpatch: int,
    Wpatch: int,
    rgb_uint8: np.ndarray,
    depth: np.ndarray | None,
    max_candidates: int,
    depth_clip: Tuple[float, float] = (0.0, 2.0),
    border_frac: float = 0.08,
    strict_depth_mask: bool = True,
) -> List[int]:
    """
    Pick candidate patches biased toward foreground/edges.
    Returns a list of patch indices into the flattened patch grid.
    """
    if max_candidates <= 0:
        raise ValueError("max_candidates must be > 0")

    H, W, _ = rgb_uint8.shape

    # Score patches by edge strength (RGB) by default.
    edge = _edge_strength_rgb(rgb_uint8)
    edge_p = _downsample_to_patch_grid_mean(edge, Hpatch, Wpatch)  # (Hpatch,Wpatch)

    score = edge_p.copy()

    # Suppress border patches (top-view often has empty/border regions).
    if border_frac > 0:
        by = int(round(border_frac * Hpatch))
        bx = int(round(border_frac * Wpatch))
        if by > 0:
            score[:by, :] = 0.0
            score[-by:, :] = 0.0
        if bx > 0:
            score[:, :bx] = 0.0
            score[:, -bx:] = 0.0

    if depth is not None:
        if depth.ndim != 2 or depth.shape[0] != H or depth.shape[1] != W:
            raise ValueError("depth must be HxW matching rgb")

        d0, d1 = depth_clip
        depth_clipped = np.clip(depth.astype(np.float32), d0, d1)
        d_norm = (depth_clipped - d0) / max(1e-6, (d1 - d0))
        d_p = _downsample_to_patch_grid_mean(d_norm, Hpatch, Wpatch)

        # Foreground heuristic: closer-than-background patches (use a more aggressive threshold)
        bg = float(np.median(d_p))
        fg = (d_p < (bg - 0.05)).astype(np.float32)

        # Depth edge heuristic
        dp = np.pad(d_p, ((1, 1), (1, 1)), mode="edge")
        gx = dp[1:-1, 2:] - dp[1:-1, :-2]
        gy = dp[2:, 1:-1] - dp[:-2, 1:-1]
        d_edge = np.sqrt(gx * gx + gy * gy)
        d_edge /= float(np.percentile(d_edge, 99) + 1e-6)
        d_edge = np.clip(d_edge, 0.0, 1.0)

        if strict_depth_mask:
            # Make depth foreground the hard candidate set. This tends to produce much more
            # object-centric keypoints for this task (cube + grippers) than global FPS.
            # If this mask is too strict for some frames, we fall back below.
            mask = fg > 0.0
            # If mask is empty, fall back to soft scoring.
            if bool(mask.any()):
                score = score + 2.0 * d_edge
                score = score * mask.astype(np.float32)
            else:
                score = score + 2.0 * d_edge + 2.0 * fg
        else:
            # Soft scoring: still allows background if it scores highly.
            score = score + 2.0 * d_edge + 2.0 * fg

    flat = score.reshape(-1)
    # If strict masking zeroed almost everything, we may have all zeros.
    # In that case, let the caller fall back by returning an empty list.
    if float(flat.max()) <= 1e-8:
        return []

    # Grab top-N candidate patches by score (stable order not required).
    n = flat.shape[0]
    max_candidates = min(max_candidates, n)
    idxs = np.argpartition(-flat, kth=max_candidates - 1)[:max_candidates]
    idxs = idxs.astype(np.int64)
    # Drop zero-score candidates so FPS can't pick junk.
    idxs = idxs[flat[idxs] > 1e-8]
    if idxs.size == 0:
        return []
    # Sort by score descending so idxs[0] is best-scoring.
    order = np.argsort(-flat[idxs])
    idxs = idxs[order]
    return idxs.tolist()


def tokenize_keypoints_2d(
    rgb_uint8: np.ndarray,
    k: int,
    device: str = "cpu",
    *,
    obs_bins: int = 64,
    max_candidates: int | None = 512,
) -> Tuple[List[Keypoint2D], str]:
    """
    Produces K normalized (x,y) keypoint tokens from a top-view RGB image.
    Returns keypoints and a token string (OBS block).
    """
    dev = torch.device(device)
    desc, Hpatch, Wpatch = extract_patch_descriptors_dinov2(rgb_uint8, device=dev)
    if max_candidates is None:
        idxs = farthest_point_sampling_cosine(desc, k=k).detach().cpu().numpy().tolist()
    else:
        cand = _choose_candidate_patch_indices(
            Hpatch=Hpatch,
            Wpatch=Wpatch,
            rgb_uint8=rgb_uint8,
            depth=None,
            max_candidates=max_candidates,
        )
        if not cand:
            idxs = farthest_point_sampling_cosine(desc, k=k).detach().cpu().numpy().tolist()
        else:
            cand_t = torch.tensor(cand, dtype=torch.long, device=desc.device)
            desc_c = desc.index_select(0, cand_t)
            # Start FPS from the best-scoring candidate (cand[0]) for stability.
            sel_local = farthest_point_sampling_cosine(
                desc_c, k=min(k, desc_c.shape[0]), start_idx=0
            )
            idxs = cand_t.index_select(0, sel_local).detach().cpu().numpy().tolist()

    H, W, _ = rgb_uint8.shape
    kps: List[Keypoint2D] = []
    for i, patch_idx in enumerate(idxs):
        x_px, y_px = patch_index_to_pixel(patch_idx, Hpatch, Wpatch, H, W)
        kp = Keypoint2D(
            x_px=x_px,
            y_px=y_px,
            x_norm=float(x_px) / float(W - 1),
            y_norm=float(y_px) / float(H - 1),
        )
        kps.append(kp)

    lines = ["OBS_START"]
    for i, kp in enumerate(kps):
        # Discrete tokens are much easier for LLMs than floats.
        xq = _quantize_01(kp.x_norm, obs_bins)
        yq = _quantize_01(kp.y_norm, obs_bins)
        lines.append(f"KP {i} x={xq} y={yq}")
    lines.append("OBS_END")
    return kps, "\n".join(lines) + "\n"


def tokenize_keypoints_2d_with_depth(
    rgb_uint8: np.ndarray,
    depth: np.ndarray,
    k: int,
    device: str = "cpu",
    depth_clip: Tuple[float, float] = (0.0, 2.0),
    *,
    obs_bins: int = 64,
    depth_bins: int | None = None,
    max_candidates: int | None = 512,
) -> Tuple[List[Keypoint2DDepth], str]:
    """
    Depth-aware extension (optional): includes a `d=` token per keypoint.

    depth is expected as HxW float (same resolution as rgb_uint8).
    """
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    if depth.shape[0] != rgb_uint8.shape[0] or depth.shape[1] != rgb_uint8.shape[1]:
        raise ValueError("depth must match rgb resolution")

    if depth_bins is None:
        depth_bins = obs_bins

    # Use the same descriptors, but bias FPS toward depth foreground/edges via candidates.
    dev = torch.device(device)
    desc, Hpatch, Wpatch = extract_patch_descriptors_dinov2(rgb_uint8, device=dev)
    if max_candidates is None:
        idxs = farthest_point_sampling_cosine(desc, k=k).detach().cpu().numpy().tolist()
    else:
        cand = _choose_candidate_patch_indices(
            Hpatch=Hpatch,
            Wpatch=Wpatch,
            rgb_uint8=rgb_uint8,
            depth=depth,
            max_candidates=max_candidates,
            depth_clip=depth_clip,
            strict_depth_mask=True,
        )
        if not cand:
            # Fall back to softer (non-strict) scoring candidates, then full-image FPS.
            cand = _choose_candidate_patch_indices(
                Hpatch=Hpatch,
                Wpatch=Wpatch,
                rgb_uint8=rgb_uint8,
                depth=depth,
                max_candidates=max_candidates,
                depth_clip=depth_clip,
                strict_depth_mask=False,
            )
        if not cand:
            idxs = farthest_point_sampling_cosine(desc, k=k).detach().cpu().numpy().tolist()
        else:
            cand_t = torch.tensor(cand, dtype=torch.long, device=desc.device)
            desc_c = desc.index_select(0, cand_t)
            sel_local = farthest_point_sampling_cosine(
                desc_c, k=min(k, desc_c.shape[0]), start_idx=0
            )
            idxs = cand_t.index_select(0, sel_local).detach().cpu().numpy().tolist()

    H, W, _ = rgb_uint8.shape
    kps2d: List[Keypoint2D] = []
    for patch_idx in idxs:
        x_px, y_px = patch_index_to_pixel(int(patch_idx), Hpatch, Wpatch, H, W)
        kps2d.append(
            Keypoint2D(
                x_px=x_px,
                y_px=y_px,
                x_norm=float(x_px) / float(W - 1),
                y_norm=float(y_px) / float(H - 1),
            )
        )

    d0, d1 = depth_clip
    depth_clipped = np.clip(depth.astype(np.float32), d0, d1)
    d_norm = (depth_clipped - d0) / max(1e-6, (d1 - d0))

    kps: List[Keypoint2DDepth] = []
    lines = ["OBS_START"]
    for i, kp in enumerate(kps2d):
        d = float(depth_clipped[kp.y_px, kp.x_px])
        dn = float(d_norm[kp.y_px, kp.x_px])
        kpd = Keypoint2DDepth(**kp.__dict__, depth=d)
        kps.append(kpd)
        xq = _quantize_01(kp.x_norm, obs_bins)
        yq = _quantize_01(kp.y_norm, obs_bins)
        dq = _quantize_01(dn, depth_bins)
        lines.append(f"KP {i} x={xq} y={yq} d={dq}")
    lines.append("OBS_END")
    return kps, "\n".join(lines) + "\n"

