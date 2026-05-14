from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from sim_env import BOX_POSE, make_sim_env
from utils import sample_box_pose


@dataclass(frozen=True)
class ReplayResult:
    success: bool
    max_reward: float
    episode_return: float
    steps: int
    video_path: Optional[str]


def upsample_waypoints_piecewise_constant(actions_wp: np.ndarray, T: int) -> np.ndarray:
    """
    actions_wp: (M,14)
    returns (T,14) by repeating each waypoint over a segment.
    """
    actions_wp = np.asarray(actions_wp, dtype=np.float32)
    if actions_wp.ndim != 2 or actions_wp.shape[1] != 14:
        raise ValueError("actions_wp must be (M,14)")
    M = actions_wp.shape[0]
    if M <= 0:
        raise ValueError("M must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")

    idxs = np.linspace(0, T, num=M + 1).round().astype(int)
    out = np.zeros((T, 14), dtype=np.float32)
    for i in range(M):
        a = actions_wp[i]
        s, e = idxs[i], idxs[i + 1]
        if e <= s:
            continue
        out[s:e] = a
    # pad any holes
    for t in range(1, T):
        if not np.any(out[t]):
            out[t] = out[t - 1]
    if not np.any(out[0]):
        out[0] = actions_wp[0]
    return out


def upsample_waypoints_linear(actions_wp: np.ndarray, T: int) -> np.ndarray:
    """
    Linearly interpolates between M waypoints to produce T actions.
    """
    actions_wp = np.asarray(actions_wp, dtype=np.float32)
    if actions_wp.ndim != 2 or actions_wp.shape[1] != 14:
        raise ValueError("actions_wp must be (M,14)")
    M = actions_wp.shape[0]
    if M <= 1:
        return np.repeat(actions_wp[:1], repeats=T, axis=0)
    if T <= 0:
        raise ValueError("T must be > 0")

    x_wp = np.linspace(0, T - 1, num=M, dtype=np.float32)
    x = np.arange(T, dtype=np.float32)
    out = np.zeros((T, 14), dtype=np.float32)
    for d in range(14):
        out[:, d] = np.interp(x, x_wp, actions_wp[:, d])
    return out


def replay_actions(
    actions: np.ndarray,
    *,
    task_name: str = "sim_transfer_cube_scripted",
    episode_len: int = 400,
    render_cam: str = "angle",
    save_video: bool = True,
    video_path: str = "artifacts/replays/replay.mp4",
    fps: int = 50,
) -> ReplayResult:
    """
    actions: (T,14) joint targets (normalized grippers), will run for min(T, episode_len).
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError("actions must be (T,14)")

    # If caller doesn't set BOX_POSE explicitly, sample a random one.
    if BOX_POSE[0] is None:
        BOX_POSE[0] = sample_box_pose()
    env = make_sim_env(task_name)
    ts = env.reset()

    out = None
    if save_video:
        h, w, _ = ts.observation["images"][render_cam].shape
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    episode = [ts]
    T_run = min(int(actions.shape[0]), int(episode_len))
    for t in range(T_run):
        ts = env.step(actions[t])
        episode.append(ts)
        if out is not None:
            img = ts.observation["images"][render_cam][:, :, [2, 1, 0]]
            out.write(img)

    if out is not None:
        out.release()

    rewards = [ts.reward for ts in episode[1:]]
    episode_return = float(np.sum(rewards))
    max_reward = float(np.max(rewards)) if len(rewards) else 0.0
    success = bool(max_reward == env.task.max_reward)

    return ReplayResult(
        success=success,
        max_reward=max_reward,
        episode_return=episode_return,
        steps=T_run,
        video_path=video_path if save_video else None,
    )

