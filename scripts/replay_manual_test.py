#!/usr/bin/env python3
"""
Replay a manually pasted LLM response on one test episode and save a video.

Examples:

  # One line (run from repo root). Do not use trailing \\ in zsh unless continuing the line.
  python3 scripts/replay_manual_test.py --tests_dir data/transfer_cube/subsets/test_5 --test_id 20 --demos_dir data/transfer_cube/subsets/demos_5 --run_id eval_icl_json_demo5_test20_idx0

  # Custom JSON file (omit --run_id if using --llm_response)
  python3 scripts/replay_manual_test.py --tests_dir data/transfer_cube/subsets/test_5 --test_id 20 --demos_dir data/transfer_cube/subsets/demos_5 --llm_response artifacts/test.json --M 20

  # Template to fill in by hand
  python scripts/replay_manual_test.py \\
    --write_template artifacts/manual_response.json --M 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from act_kat.episodes import episode_path, list_episode_ids, read_env_state0
from act_kat.icl import (
    build_actions_json_schema,
    fit_quantizer_from_episodes,
    read_llm_response_file,
    replay_from_icl_text,
)
from act_kat.ollama_client import response_json_path


def _write_template(out_path: str, M: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    template = {
        "actions": [
            {"L": [0] * 6, "LG": 0, "R": [0] * 6, "RG": 0}
        ]
        * M
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print(f"Wrote template ({M} waypoints): {out_path}")
    print("Edit L/R (6 ints each), LG/RG in {0,1}, then replay with --llm_response", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay manual LLM JSON on one test episode; save rollout video.",
    )
    ap.add_argument("--tests_dir", type=str, default=None)
    ap.add_argument("--test_id", type=int, default=None)
    ap.add_argument("--test_episode_hdf5", type=str, default=None)
    ap.add_argument("--demos_dir", type=str, default=None)
    ap.add_argument("--quantizer_json", type=str, default=None)
    ap.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Shorthand: load artifacts/responses/<run_id>.json",
    )
    ap.add_argument(
        "--llm_response",
        "--llm_json",
        dest="llm_response",
        type=str,
        default=None,
        help="Path to response .json (or legacy .txt), or '-' for stdin",
    )
    ap.add_argument("--llm_json_text", type=str, default=None)
    ap.add_argument(
        "--save_llm_response",
        type=str,
        default=None,
        help="Optional path to copy the raw LLM text",
    )
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument(
        "--upsample",
        type=str,
        default="linear",
        choices=["linear", "hold"],
    )
    ap.add_argument("--video_path", type=str, default=None)
    ap.add_argument("--write_template", type=str, default=None, metavar="PATH")
    args = ap.parse_args()

    if args.write_template:
        _write_template(args.write_template, args.M or 20)
        return

    llm_path = args.llm_response
    if args.run_id:
        if llm_path is not None:
            ap.error("use only one of --run_id and --llm_response")
        llm_path = response_json_path("artifacts", args.run_id)
        if not os.path.isfile(llm_path):
            legacy_txt = llm_path.replace(".json", ".txt")
            if os.path.isfile(legacy_txt):
                llm_path = legacy_txt
            else:
                raise SystemExit(f"response file not found: {llm_path}")

    if llm_path is None and args.llm_json_text is None:
        ap.error("provide --run_id, --llm_response, --llm_json_text, or --write_template")

    if args.test_episode_hdf5:
        test_path = args.test_episode_hdf5
        test_id = None
        for part in os.path.basename(test_path).replace(".hdf5", "").split("_"):
            if part.isdigit():
                test_id = int(part)
                break
    else:
        if args.tests_dir is None or args.test_id is None:
            ap.error("provide --tests_dir and --test_id, or --test_episode_hdf5")
        test_path = episode_path(args.tests_dir, args.test_id)
        test_id = args.test_id
        if test_id not in list_episode_ids(args.tests_dir):
            raise SystemExit(f"test_id {test_id} not in {args.tests_dir}")

    if args.quantizer_json:
        with open(args.quantizer_json, "r", encoding="utf-8") as f:
            qj = json.load(f)
        from act_kat.action_tokens import ActionQuantizer
        import numpy as np

        quant = ActionQuantizer(
            mins=np.asarray(qj["mins"], dtype=np.float32),
            maxs=np.asarray(qj["maxs"], dtype=np.float32),
            bins=args.bins,
        )
    elif args.demos_dir:
        demo_ids = list_episode_ids(args.demos_dir)
        if not demo_ids:
            raise SystemExit(f"No episodes in demos_dir: {args.demos_dir}")
        quant = fit_quantizer_from_episodes(
            [episode_path(args.demos_dir, ep) for ep in demo_ids], bins=args.bins
        )
    else:
        ap.error("provide --demos_dir or --quantizer_json")

    if args.llm_json_text is not None:
        llm_text = args.llm_json_text
    elif llm_path == "-":
        llm_text = sys.stdin.read()
    else:
        llm_text = read_llm_response_file(llm_path)

    if args.save_llm_response:
        os.makedirs(os.path.dirname(args.save_llm_response) or ".", exist_ok=True)
        with open(args.save_llm_response, "w", encoding="utf-8") as f:
            f.write(llm_text.strip())
        print(f"[saved_llm] {args.save_llm_response}")

    video_path = args.video_path or (
        f"artifacts/replays/manual_test{test_id}.mp4"
        if test_id is not None
        else "artifacts/replays/manual_episode.mp4"
    )
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    with h5py.File(test_path, "r") as root:
        env0 = read_env_state0(root)

    res = replay_from_icl_text(
        llm_text,
        quant,
        bins=args.bins,
        episode_len=args.episode_len,
        env_state0=env0,
        video_path=video_path,
        M=args.M,
        upsample=args.upsample,
    )

    print(
        json.dumps(
            {
                "test_episode": test_path,
                "llm_response": llm_path,
                "video_path": res.video_path,
                "success": res.success,
                "max_reward": res.max_reward,
                "steps": res.steps,
            },
            indent=2,
        )
    )
    if not res.success:
        print(f"(max_reward={res.max_reward}; need 4 for transfer cube success)")


if __name__ == "__main__":
    main()
