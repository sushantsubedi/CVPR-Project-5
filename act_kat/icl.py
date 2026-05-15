"""KAT-style in-context imitation: tokenize, prompt, generate, parse, replay."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode
from act_kat.episodes import read_top_frame
from act_kat.keypoint_anchors import (
    anchor_descriptors_from_episode,
    tokenize_anchored_keypoints,
)
from act_kat.ollama_client import ollama_generate
from act_kat.replay import ReplayResult, replay_actions, upsample_waypoints


def tokenize_obs(
    img: np.ndarray,
    *,
    anchor_desc: np.ndarray,
    depth: Optional[np.ndarray] = None,
    device: str = "cpu",
    obs_bins: int = 16,
) -> str:
    """DINO-anchored keypoint OBS block (paper-style)."""
    _, tok = tokenize_anchored_keypoints(
        img, anchor_desc, depth=depth, device=device, obs_bins=obs_bins
    )
    return tok


def fit_quantizer_from_episodes(
    episode_paths: List[str], bins: int
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
    anchor_desc: np.ndarray,
    device: str = "cpu",
    obs_bins: int = 16,
) -> List[str]:
    pairs: List[str] = []
    for path in demo_paths:
        with h5py.File(path, "r") as root:
            img, depth = read_top_frame(root, t=0)
            act = root["/action"][()].astype(np.float32)
        obs_tok = tokenize_obs(
            img,
            anchor_desc=anchor_desc,
            depth=depth,
            device=device,
            obs_bins=obs_bins,
        )
        _, act_tok = action_tokens_from_episode(act, quantizer=quant, M=M)
        pairs.append(obs_tok + act_tok)
    return pairs


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


def build_prompt(demo_pairs: List[str], query_obs: str, M: int) -> str:
    parts = ["### Demonstrations"]
    for i, pair in enumerate(demo_pairs):
        parts.append(f"## Demo {i}")
        parts.append(pair.strip())
        parts.append("")
    parts.append("### Query")
    parts.append(query_obs.strip())
    parts.append(
        "Predict the query trajectory as JSON (M waypoints: L, LG, R, RG). "
        "OBS is K DINO keypoints (stable index i, quantized x/y/d). "
        "Demos use TRAJ/WP lines with the same L, LG, R, RG fields (LG/RG: 0=open, 1=closed)."
    )
    parts.append(
        "\nReturn ONLY JSON matching the provided schema (not TRAJ text). "
        f"Output exactly M={M} waypoint objects with keys L, LG, R, RG. "
        "L/R = length-6 integer joint code arrays."
    )
    return "\n".join(parts) + "\n"


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
    prompt = build_prompt(demo_pairs, query_obs, M)
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


def load_actions_payload(text: str) -> List[Dict[str, Any]]:
    """Parse LLM output into the list of {L, LG, R, RG} waypoints."""
    text = text.strip()
    if not text:
        raise ValueError("empty response text")
    data = json.loads(text)
    if isinstance(data, dict) and "actions" in data:
        actions = data["actions"]
    elif isinstance(data, dict) and "response" in data:
        inner = data["response"]
        inner_data = json.loads(inner) if isinstance(inner, str) else inner
        actions = inner_data.get("actions")
    else:
        raise ValueError('expected JSON with "actions" key')
    if not isinstance(actions, list) or not actions:
        raise ValueError("missing or empty actions array")
    return actions


def parse_actions_strict(text: str, M: int) -> List[Dict[str, Any]]:
    actions = load_actions_payload(text)
    if len(actions) != M:
        raise ValueError(f"expected {M} actions, got {len(actions)}")
    return actions


def actions_to_codes(actions: List[Dict[str, Any]], bins: int) -> np.ndarray:
    codes = np.zeros((len(actions), 14), dtype=np.int32)
    for i, a in enumerate(actions):
        L, R = a["L"], a["R"]
        if not (isinstance(L, list) and len(L) == 6 and isinstance(R, list) and len(R) == 6):
            raise ValueError(f"bad L/R at i={i}")
        codes[i, :6] = [int(x) for x in L]
        codes[i, 7:13] = [int(x) for x in R]
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
    return upsample_waypoints(
        quant.decode(codes).astype(np.float32), T=episode_len, mode=upsample
    )


def replay_response(
    text: str,
    quant: ActionQuantizer,
    *,
    env_state0: np.ndarray,
    video_path: str,
    bins: int,
    episode_len: int = 400,
    M: Optional[int] = None,
    upsample: str = "linear",
) -> ReplayResult:
    """Decode an LLM response and replay it in the sim."""
    from sim_env import BOX_POSE

    actions = parse_actions_strict(text, M) if M else load_actions_payload(text)
    codes = actions_to_codes(actions, bins=bins)
    traj = codes_to_trajectory(codes, quant, episode_len=episode_len, upsample=upsample)
    BOX_POSE[0] = env_state0
    return replay_actions(traj, episode_len=episode_len, video_path=video_path)


__all__ = [
    "tokenize_obs",
    "build_demo_pairs",
    "fit_quantizer_from_episodes",
    "anchor_descriptors_from_episode",
    "build_actions_json_schema",
    "build_prompt",
    "generate_icl_actions",
    "load_actions_payload",
    "parse_actions_strict",
    "actions_to_codes",
    "codes_to_trajectory",
    "replay_response",
]
