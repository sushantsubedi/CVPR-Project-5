from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class OllamaRequestLog:
    created_at_utc: str
    url: str
    model: str
    options: Dict[str, Any]
    stream: bool
    prompt_path: str


@dataclass
class OllamaResponseLog:
    created_at_utc: str
    url: str
    model: str
    status_code: int
    elapsed_s: float
    response_path: str


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
      - artifacts/prompts/<run_id>.txt
      - artifacts/ollama_requests/<run_id>.json
      - artifacts/ollama_responses/<run_id>.json

    Returns the parsed JSON response.
    """
    if options is None:
        options = {}
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompts_dir = os.path.join(artifacts_dir, "prompts")
    req_dir = os.path.join(artifacts_dir, "ollama_requests")
    resp_dir = os.path.join(artifacts_dir, "ollama_responses")
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    prompt_path = os.path.join(prompts_dir, f"{run_id}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if format is not None:
        payload["format"] = format

    req_log = OllamaRequestLog(
        created_at_utc=_utc_now_iso(),
        url=url,
        model=model,
        options=options,
        stream=False,
        prompt_path=prompt_path,
    )
    with open(os.path.join(req_dir, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(req_log) | {"payload": payload}, f, indent=2)

    t0 = time.time()
    r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout_s)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    resp_path = os.path.join(resp_dir, f"{run_id}.json")
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    resp_log = OllamaResponseLog(
        created_at_utc=_utc_now_iso(),
        url=url,
        model=model,
        status_code=r.status_code,
        elapsed_s=float(elapsed),
        response_path=resp_path,
    )
    with open(os.path.join(resp_dir, f"{run_id}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(resp_log), f, indent=2)

    return data


def ollama_chat(
    *,
    system: str,
    user: str,
    model: str,
    url: str = "http://127.0.0.1:11434",
    options: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    timeout_s: int = 600,
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls Ollama /api/chat (non-stream) and logs the request/response.
    Some models may populate `message.thinking` instead of `message.content`;
    downstream code should handle that.
    """
    if options is None:
        options = {}
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompts_dir = os.path.join(artifacts_dir, "prompts")
    req_dir = os.path.join(artifacts_dir, "ollama_requests")
    resp_dir = os.path.join(artifacts_dir, "ollama_responses")
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    prompt_path = os.path.join(prompts_dir, f"{run_id}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(system.strip() + "\n\n" + user)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
    }
    if format is not None:
        payload["format"] = format

    req_log = OllamaRequestLog(
        created_at_utc=_utc_now_iso(),
        url=url,
        model=model,
        options=options,
        stream=False,
        prompt_path=prompt_path,
    )
    with open(os.path.join(req_dir, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(req_log) | {"payload": payload}, f, indent=2)

    t0 = time.time()
    r = requests.post(f"{url}/api/chat", json=payload, timeout=timeout_s)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    resp_path = os.path.join(resp_dir, f"{run_id}.json")
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    resp_log = OllamaResponseLog(
        created_at_utc=_utc_now_iso(),
        url=url,
        model=model,
        status_code=r.status_code,
        elapsed_s=float(elapsed),
        response_path=resp_path,
    )
    with open(os.path.join(resp_dir, f"{run_id}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(resp_log), f, indent=2)

    return data

