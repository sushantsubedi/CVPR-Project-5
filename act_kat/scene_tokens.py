"""TRAJ / WP formatting for demo action blocks (OBS uses keypoint_anchors)."""
from __future__ import annotations

import json

import numpy as np


def format_action_trajectory(
    q_waypoints: np.ndarray,
    *,
    lg: np.ndarray,
    rg: np.ndarray,
) -> str:
    """
    Format (M,14) quantized waypoint codes as TRAJ / WP lines.

    Uses the same field names and gripper values as JSON output: L, LG, R, RG
    with LG/RG in {0, 1} (0=open, 1=closed).
    """
    q = np.asarray(q_waypoints, dtype=np.int32)
    lg = np.asarray(lg, dtype=np.int32).reshape(-1)
    rg = np.asarray(rg, dtype=np.int32).reshape(-1)
    M = q.shape[0]
    lines = ["TRAJ_START"]
    for i in range(M):
        lines.append(
            f"WP[i={i}] L={json.dumps(q[i, :6].tolist())} LG={int(lg[i])} "
            f"R={json.dumps(q[i, 7:13].tolist())} RG={int(rg[i])}"
        )
    lines.append("TRAJ_END")
    return "\n".join(lines) + "\n"
