from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode
from act_kat.episodes import episode_path, list_episode_ids, read_env_state0, read_top_frame
from act_kat.ollama_client import ollama_generate
from act_kat.prompting import build_chat_user_message
from act_kat.replay import ReplayResult, replay_actions, upsample_waypoints
from act_kat.keypoint_anchors import (  # noqa: F401 re-export
    anchor_descriptors_from_episode,
    tokenize_anchored_keypoints,
)
from act_kat.vision_tokens import tokenize_keypoints_2d, tokenize_keypoints_2d_with_depth


def tokenize_obs(
    img: np.ndarray,
    *,
    depth: Optional[np.ndarray] = None,
    k: int,
    device: str = "cpu",
    obs_bins: int = 64,
    anchor_desc: Optional[np.ndarray] = None,
    keypoint_mode: str = "anchored",
    token_format: str = "keypoints",
) -> str:
    """
    token_format:
      - keypoints (default): paper-style DINO anchors + NN localize → OBS_START / KP i x y [d]
      - legacy: same blocks, but keypoint_mode can select fps instead of anchored
    keypoint_mode: anchored (shared descriptors across episodes) | fps (per-frame FPS)
    """
    use_anchored = token_format == "keypoints" or keypoint_mode == "anchored"
    if use_anchored:
        if anchor_desc is None:
            raise ValueError("anchored mode requires anchor_desc; call anchor_descriptors_from_episode first")
        _, tok = tokenize_anchored_keypoints(
            img, anchor_desc, depth=depth, device=device, obs_bins=obs_bins
        )
        return tok
    if token_format == "legacy" and keypoint_mode == "fps":
        if depth is not None and np.isfinite(np.asarray(depth)).any():
            _, tok = tokenize_keypoints_2d_with_depth(
                img, depth=depth, k=k, device=device, obs_bins=obs_bins
            )
        else:
            _, tok = tokenize_keypoints_2d(img, k=k, device=device, obs_bins=obs_bins)
        return tok
    raise ValueError(
        f"unknown token_format/keypoint_mode: {token_format!r} / {keypoint_mode!r}"
    )


def build_actions_json_schema(M: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": M,
                "maxItems": M,
                "items": {
                    "type": "object",
                    "properties": {
                        "L": {
                            "type": "array",
                            "minItems": 6,
                            "maxItems": 6,
                            "items": {"type": "integer"},
                        },
                        "LG": {"type": "integer", "enum": [0, 1]},
                        "R": {
                            "type": "array",
                            "minItems": 6,
                            "maxItems": 6,
                            "items": {"type": "integer"},
                        },
                        "RG": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["L", "LG", "R", "RG"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def read_llm_response_file(path: str) -> str:
    """Load a saved LLM response (.json, legacy .txt, or full Ollama API dump)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_icl_actions_payload(text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM output into the actions list.

    Accepts:
      - raw JSON: {"actions": [...]}
      - legacy Ollama API dump: {"response": "<json string>", ...}
    """
    text = text.strip()
    if not text:
        raise ValueError("empty JSON text")

    data = json.loads(text)
    if isinstance(data, dict) and "actions" in data:
        actions = data["actions"]
    elif isinstance(data, dict) and "response" in data:
        inner = data["response"]
        if isinstance(inner, str):
            inner_data = json.loads(inner.strip())
        elif isinstance(inner, dict):
            inner_data = inner
        else:
            raise ValueError("ollama response field must be str or dict")
        actions = inner_data.get("actions")
    else:
        raise ValueError('expected JSON with "actions" or Ollama {"response": ...}')

    if not isinstance(actions, list) or len(actions) == 0:
        raise ValueError("expected non-empty actions list")
    return actions


def parse_icl_actions_json(text: str, M: int) -> List[Dict[str, Any]]:
    actions = load_icl_actions_payload(text)
    if len(actions) != M:
        raise ValueError(f"expected {M} actions, got {len(actions)}")
    return actions


def actions_json_to_codes(actions: List[Dict[str, Any]], bins: int) -> np.ndarray:
    codes = np.zeros((len(actions), 14), dtype=np.int32)
    for i, a in enumerate(actions):
        L = a["L"]
        R = a["R"]
        if not (isinstance(L, list) and len(L) == 6 and isinstance(R, list) and len(R) == 6):
            raise ValueError(f"bad L/R at i={i}")
        codes[i, :6] = np.asarray([int(x) for x in L], dtype=np.int32)
        codes[i, 7:13] = np.asarray([int(x) for x in R], dtype=np.int32)
        codes[i, 6] = int(a["LG"]) * (bins - 1)
        codes[i, 13] = int(a["RG"]) * (bins - 1)
    return codes


def codes_to_trajectory(
    codes: np.ndarray,
    quant: ActionQuantizer,
    *,
    episode_len: int,
    upsample: str = "linear",
) -> np.ndarray:
    actions_wp = quant.decode(codes).astype(np.float32)
    return upsample_waypoints(actions_wp, T=episode_len, mode=upsample)


def fit_quantizer_from_episodes(
    episode_paths: List[str],
    bins: int,
) -> ActionQuantizer:
    chunks = []
    for path in episode_paths:
        with h5py.File(path, "r") as root:
            chunks.append(root["/action"][()].astype(np.float32))
    actions = np.concatenate(chunks, axis=0)
    return ActionQuantizer(
        mins=actions.min(axis=0), maxs=actions.max(axis=0), bins=bins
    )


def build_demo_pairs(
    demo_paths: List[str],
    quant: ActionQuantizer,
    *,
    K: int,
    M: int,
    device: str,
    obs_bins: int,
    anchor_desc: Optional[np.ndarray] = None,
    keypoint_mode: str = "anchored",
    token_format: str = "scene",
) -> List[str]:
    if keypoint_mode == "anchored" and anchor_desc is None:
        anchor_desc = anchor_descriptors_from_episode(demo_paths[0], K, device=device)
    pairs: List[str] = []
    for path in demo_paths:
        with h5py.File(path, "r") as root:
            img, depth = read_top_frame(root, t=0)
            act = root["/action"][()].astype(np.float32)
        obs_tok = tokenize_obs(
            img,
            depth=depth,
            k=K,
            device=device,
            obs_bins=obs_bins,
            anchor_desc=anchor_desc,
            keypoint_mode=keypoint_mode,
            token_format=token_format,
        )
        _, act_tok = action_tokens_from_episode(
            act, quantizer=quant, M=M, token_format=token_format
        )
        pairs.append(obs_tok + act_tok)
    return pairs


def icl_json_prompt_suffix(M: int) -> str:
    return (
        "\n\nReturn ONLY JSON matching the provided schema (not TRAJ text).\n"
        f"Output exactly M={M} waypoint objects with keys L, LG, R, RG.\n"
        "Same as demo WP lines: L/R = length-6 joint code arrays; LG/RG = 0 (open) or 1 (closed).\n"
    )


def generate_icl_actions(
    *,
    demo_pairs: List[str],
    query_obs: str,
    M: int,
    model: str,
    ollama_url: str,
    run_id: str,
    artifacts_dir: str = "artifacts",
    timeout_s: int = 600,
) -> str:
    prompt = build_chat_user_message(demo_pairs, query_obs=query_obs) + icl_json_prompt_suffix(M)
    resp = ollama_generate(
        prompt=prompt,
        model=model,
        url=ollama_url,
        options={"temperature": 0.0, "num_predict": 2048},
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        format=build_actions_json_schema(M),
        timeout_s=timeout_s,
    )
    return resp.get("response", "") if isinstance(resp, dict) else ""


def save_llm_response_json(text: str, run_id: str, artifacts_dir: str = "artifacts") -> str:
    from act_kat.ollama_client import response_json_path, write_response_json

    out_path = response_json_path(artifacts_dir, run_id)
    write_response_json(out_path, text)
    return out_path


def replay_icl_actions(
    actions: List[Dict[str, Any]],
    quantizer: ActionQuantizer,
    *,
    bins: int,
    episode_len: int,
    env_state0: np.ndarray,
    video_path: str,
    upsample: str = "linear",
) -> ReplayResult:
    from sim_env import BOX_POSE

    codes = actions_json_to_codes(actions, bins=bins)
    trajectory = codes_to_trajectory(
        codes, quantizer, episode_len=episode_len, upsample=upsample
    )
    BOX_POSE[0] = env_state0
    return replay_actions(trajectory, episode_len=episode_len, video_path=video_path)


def replay_from_icl_text(
    icl_text: str,
    quantizer: ActionQuantizer,
    *,
    bins: int,
    episode_len: int,
    env_state0: np.ndarray,
    video_path: str,
    M: Optional[int] = None,
    upsample: str = "linear",
) -> ReplayResult:
    actions = (
        parse_icl_actions_json(icl_text, M)
        if M is not None
        else load_icl_actions_payload(icl_text)
    )
    return replay_icl_actions(
        actions,
        quantizer,
        bins=bins,
        episode_len=episode_len,
        env_state0=env_state0,
        video_path=video_path,
        upsample=upsample,
    )


def replay_from_icl_json_file(
    icl_json_path: str,
    quantizer: ActionQuantizer,
    *,
    bins: int,
    episode_len: int,
    env_state0: np.ndarray,
    video_path: str,
    M: Optional[int] = None,
    upsample: str = "linear",
) -> ReplayResult:
    with open(icl_json_path, "r", encoding="utf-8") as f:
        return replay_from_icl_text(
            f.read(),
            quantizer,
            bins=bins,
            episode_len=episode_len,
            env_state0=env_state0,
            video_path=video_path,
            M=M,
            upsample=upsample,
        )


__all__ = [
    "list_episode_ids",
    "episode_path",
    "tokenize_obs",
    "build_actions_json_schema",
    "load_icl_actions_payload",
    "parse_icl_actions_json",
    "replay_icl_actions",
    "replay_from_icl_text",
    "actions_json_to_codes",
    "codes_to_trajectory",
    "fit_quantizer_from_episodes",
    "build_demo_pairs",
    "generate_icl_actions",
    "read_llm_response_file",
    "save_llm_response_json",
    "replay_from_icl_json_file",
]
