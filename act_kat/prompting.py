from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptConfig:
    task_name: str = "sim_transfer_cube_scripted"
    model_name: str = "gemma4:26b"
    k_keypoints: int = 10
    m_actions: int = 20


SYSTEM_INSTRUCTION = """You are controlling a simulated bimanual robot.\nYou will be given demonstrations mapping OBS tokens to ACT tokens.\n\nCRITICAL RESPONSE FORMAT:\n- Your response MUST start with the exact line: ACT_START\n- Then output only lines that match: A i=<int> L=[..6 ints..] LG=<0|1> R=[..6 ints..] RG=<0|1>\n- Then output the exact line: ACT_END\n- Do not output any other text, reasoning, markdown, or blank lines.\n"""


def build_fewshot_prompt(demo_pairs: List[str], query_obs: str) -> str:
    """
    demo_pairs: list of strings, each already formatted as:
      OBS...\nACT...\n
    query_obs: OBS block for query episode
    """
    parts: List[str] = []
    parts.append("### Demonstrations")
    for i, pair in enumerate(demo_pairs):
        parts.append(f"## Demo {i}")
        parts.append(pair.strip())
        parts.append("")
    parts.append("### Query")
    parts.append(query_obs.strip())
    parts.append("Return ACT for the query OBS.")
    return "\n".join(parts) + "\n"


def build_chat_user_message(demo_pairs: List[str], query_obs: str) -> str:
    # Same prompt body but without the system header.
    return build_fewshot_prompt(demo_pairs, query_obs=query_obs)

