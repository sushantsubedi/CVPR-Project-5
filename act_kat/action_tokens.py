"""14-D bimanual action quantization and TRAJ / WP token formatting."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ActionQuantizer:
    """Per-dimension uniform quantizer for 14-D joint+gripper actions."""

    mins: np.ndarray  # (14,)
    maxs: np.ndarray  # (14,)
    bins: int = 64

    def __post_init__(self) -> None:
        if self.mins.shape != (14,) or self.maxs.shape != (14,):
            raise ValueError("mins/maxs must be shape (14,)")
        if not (2 <= self.bins <= 4096):
            raise ValueError("bins must be in [2, 4096]")

    def encode(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        if a.shape[-1] != 14:
            raise ValueError("action last dim must be 14")
        denom = np.maximum(self.maxs - self.mins, 1e-6)
        x = np.clip((a - self.mins) / denom, 0.0, 1.0)
        return np.floor(x * (self.bins - 1) + 0.5).astype(np.int32)

    def decode(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.int32)
        if q.shape[-1] != 14:
            raise ValueError("codes last dim must be 14")
        q = np.clip(q, 0, self.bins - 1).astype(np.float32)
        x = q / float(self.bins - 1)
        return (self.mins + x * (self.maxs - self.mins)).astype(np.float32)


def select_waypoint_indices(T: int, M: int) -> np.ndarray:
    if T <= 0 or not (1 <= M <= T):
        raise ValueError(f"need T>0 and 1<=M<=T, got T={T} M={M}")
    return np.linspace(0, T - 1, num=M).round().astype(np.int32)


def format_action_trajectory(
    q_waypoints: np.ndarray, *, lg: np.ndarray, rg: np.ndarray
) -> str:
    """(M,14) int codes -> TRAJ_START / WP[i=...] / TRAJ_END block.

    Field names mirror JSON output: L, LG, R, RG with LG/RG in {0,1}.
    """
    q = np.asarray(q_waypoints, dtype=np.int32)
    lg = np.asarray(lg, dtype=np.int32).reshape(-1)
    rg = np.asarray(rg, dtype=np.int32).reshape(-1)
    lines = ["TRAJ_START"]
    for i in range(q.shape[0]):
        lines.append(
            f"WP[i={i}] L={json.dumps(q[i, :6].tolist())} LG={int(lg[i])} "
            f"R={json.dumps(q[i, 7:13].tolist())} RG={int(rg[i])}"
        )
    lines.append("TRAJ_END")
    return "\n".join(lines) + "\n"


def action_tokens_from_episode(
    actions: np.ndarray,
    quantizer: ActionQuantizer,
    M: int,
    gripper_threshold: float = 0.5,
) -> Tuple[np.ndarray, str]:
    """Subsample M waypoints from (T,14) actions; return codes + token block."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError("actions must be (T,14)")
    idxs = select_waypoint_indices(actions.shape[0], M)
    a_wp = actions[idxs]
    q = quantizer.encode(a_wp)
    lg = (a_wp[:, 6] >= gripper_threshold).astype(np.int32)
    rg = (a_wp[:, 13] >= gripper_threshold).astype(np.int32)
    return q, format_action_trajectory(q, lg=lg, rg=rg)
