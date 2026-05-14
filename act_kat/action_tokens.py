from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ActionQuantizer:
    """
    Per-dimension uniform quantizer for 14D ACT sim joint+gripper action vectors.
    """

    mins: np.ndarray  # (14,)
    maxs: np.ndarray  # (14,)
    bins: int = 64

    def __post_init__(self) -> None:
        if self.mins.shape != (14,) or self.maxs.shape != (14,):
            raise ValueError("mins/maxs must be shape (14,)")
        if not (2 <= self.bins <= 4096):
            raise ValueError("bins must be reasonable")

    def encode(self, a: np.ndarray) -> np.ndarray:
        """a: (..., 14) -> int codes (..., 14) in [0, bins-1]"""
        a = np.asarray(a, dtype=np.float32)
        if a.shape[-1] != 14:
            raise ValueError("action last dim must be 14")
        denom = np.maximum(self.maxs - self.mins, 1e-6)
        x = (a - self.mins) / denom
        x = np.clip(x, 0.0, 1.0)
        q = np.floor(x * (self.bins - 1) + 0.5).astype(np.int32)
        return q

    def decode(self, q: np.ndarray) -> np.ndarray:
        """q: (...,14) int -> float (...,14)"""
        q = np.asarray(q, dtype=np.int32)
        if q.shape[-1] != 14:
            raise ValueError("codes last dim must be 14")
        q = np.clip(q, 0, self.bins - 1).astype(np.float32)
        x = q / float(self.bins - 1)
        return (self.mins + x * (self.maxs - self.mins)).astype(np.float32)


def build_quantizer_from_dataset(
    dataset_dir: str,
    episode_ids: List[int],
    bins: int = 64,
) -> ActionQuantizer:
    """
    Computes mins/maxs over provided episodes (per dimension).
    """
    import h5py  # local import to keep module light
    mins = np.full((14,), np.inf, dtype=np.float32)
    maxs = np.full((14,), -np.inf, dtype=np.float32)
    for ep in episode_ids:
        path = f"{dataset_dir}/episode_{ep}.hdf5"
        with h5py.File(path, "r") as root:
            a = root["/action"][()].astype(np.float32)
        mins = np.minimum(mins, a.min(axis=0))
        maxs = np.maximum(maxs, a.max(axis=0))
    return ActionQuantizer(mins=mins, maxs=maxs, bins=bins)


def select_waypoint_indices(T: int, M: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be > 0")
    if not (1 <= M <= T):
        raise ValueError("M must be in [1,T]")
    return np.linspace(0, T - 1, num=M).round().astype(np.int32)


def action_tokens_from_episode(
    actions: np.ndarray,
    quantizer: ActionQuantizer,
    M: int,
    gripper_threshold: float = 0.5,
) -> Tuple[np.ndarray, str]:
    """
    Returns:
      q_waypoints: (M,14) int codes
      token_block: string with ACT_START/ACT_END
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError("actions must be (T,14)")
    T = actions.shape[0]
    idxs = select_waypoint_indices(T, M)
    a_wp = actions[idxs]
    q = quantizer.encode(a_wp)  # (M,14)

    # Optional: binarize grippers (dims 6 and 13), but still keep quantized value for decoding.
    lg = (a_wp[:, 6] >= gripper_threshold).astype(np.int32)
    rg = (a_wp[:, 13] >= gripper_threshold).astype(np.int32)

    lines = ["ACT_START"]
    for i in range(M):
        left = q[i, :6].tolist()
        right = q[i, 7:13].tolist()
        lines.append(f"A i={i} L={json.dumps(left)} LG={int(lg[i])} R={json.dumps(right)} RG={int(rg[i])}")
    lines.append("ACT_END")
    return q, "\n".join(lines) + "\n"


_LINE_RE = re.compile(
    r"^A\s+i=(?P<i>\d+)\s+L=(?P<L>\[[0-9,\s]+\])\s+LG=(?P<LG>[01])\s+R=(?P<R>\[[0-9,\s]+\])\s+RG=(?P<RG>[01])\s*$"
)


def parse_action_block(text: str, M_expected: Optional[int] = None) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Parses ACT block into integer codes (M,14).
    Returns (codes_or_none, errors).
    """
    errors: List[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if "ACT_START" not in lines or "ACT_END" not in lines:
        errors.append("missing_ACT_START_or_END")
        return None, errors
    try:
        start = lines.index("ACT_START")
        end = lines.index("ACT_END")
    except ValueError:
        errors.append("bad_ACT_markers")
        return None, errors
    body = lines[start + 1 : end]
    if len(body) == 0:
        errors.append("empty_ACT_body")
        return None, errors

    parsed: Dict[int, Tuple[List[int], int, List[int], int]] = {}
    for ln in body:
        m = _LINE_RE.match(ln)
        if not m:
            errors.append(f"unparsed_line:{ln[:80]}")
            continue
        i = int(m.group("i"))
        try:
            L = json.loads(m.group("L"))
            R = json.loads(m.group("R"))
        except Exception:
            errors.append(f"bad_json:{i}")
            continue
        if not (isinstance(L, list) and len(L) == 6 and isinstance(R, list) and len(R) == 6):
            errors.append(f"bad_lengths:{i}")
            continue
        LG = int(m.group("LG"))
        RG = int(m.group("RG"))
        parsed[i] = ([int(x) for x in L], LG, [int(x) for x in R], RG)

    if len(parsed) == 0:
        errors.append("no_valid_lines")
        return None, errors

    M = max(parsed.keys()) + 1
    if M_expected is not None and M != M_expected:
        errors.append(f"M_mismatch:got={M},expected={M_expected}")

    q = np.zeros((M, 14), dtype=np.int32)
    for i, (L, LG, R, RG) in parsed.items():
        q[i, :6] = np.asarray(L, dtype=np.int32)
        # store gripper bits in dims 6 and 13; caller can map to bin extremes if desired
        q[i, 6] = int(LG)
        q[i, 7:13] = np.asarray(R, dtype=np.int32)
        q[i, 13] = int(RG)
    return q, errors

