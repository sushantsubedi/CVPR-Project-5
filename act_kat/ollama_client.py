from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests


def response_json_path(artifacts_dir: str, run_id: str) -> str:
    return os.path.join(artifacts_dir, "responses", f"{run_id}.json")


def prompt_txt_path(artifacts_dir: str, run_id: str) -> str:
    return os.path.join(artifacts_dir, "prompts", f"{run_id}.txt")


def write_response_json(path: str, response_text: str) -> None:
    """Write model output as formatted JSON when parseable."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = response_text.strip()
    try:
        data = json.loads(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except json.JSONDecodeError:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def ollama_generate(
    *,
    prompt: str,
    model: str,
    url: str = "http://127.0.0.1:11434",
    options: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    timeout_s: int = 600,
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls Ollama /api/generate (non-stream) and logs:
      - artifacts/prompts/<run_id>.txt    — full prompt
      - artifacts/responses/<run_id>.json — parsed actions JSON (pretty-printed)

    Returns the parsed Ollama API JSON (for callers that need it).
    """
    if options is None:
        options = {}
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(os.path.join(artifacts_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "responses"), exist_ok=True)

    with open(prompt_txt_path(artifacts_dir, run_id), "w", encoding="utf-8") as f:
        f.write(prompt)

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if format is not None:
        payload["format"] = format

    r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    response_text = data.get("response", "") if isinstance(data, dict) else ""
    if not isinstance(response_text, str):
        response_text = json.dumps(response_text)

    write_response_json(response_json_path(artifacts_dir, run_id), response_text)
    return data
