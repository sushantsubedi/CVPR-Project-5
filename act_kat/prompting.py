from __future__ import annotations

from typing import List


def build_fewshot_prompt(demo_pairs: List[str], query_obs: str) -> str:
    parts: List[str] = ["### Demonstrations"]
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
    return "\n".join(parts) + "\n"


def build_chat_user_message(demo_pairs: List[str], query_obs: str) -> str:
    return build_fewshot_prompt(demo_pairs, query_obs=query_obs)
