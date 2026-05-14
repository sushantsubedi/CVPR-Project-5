import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer, parse_action_block
from act_kat.replay import replay_actions, upsample_waypoints_piecewise_constant
from sim_env import BOX_POSE


def extract_text_from_ollama_response(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "response" in data:
        return data.get("response") or ""
    msg = data.get("message") or {}
    if isinstance(msg, dict):
        return (msg.get("content") or "") + "\n" + (msg.get("thinking") or "")
    return ""


def extract_act_block(text: str) -> str:
    import re

    # Find an ACT block where ACT_START/ACT_END appear at start-of-line (allow leading spaces).
    pattern = re.compile(r"(?ms)^[ \t]*ACT_START[ \t]*\n.*?^[ \t]*ACT_END[ \t]*$")
    m = pattern.search(text)
    if not m:
        return ""
    return m.group(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama_response_json", type=str, required=True)
    ap.add_argument("--quantizer_json", type=str, required=True, help="Path to quantizer mins/maxs json")
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--video_path", type=str, default="artifacts/replays/replay_from_ollama.mp4")
    ap.add_argument(
        "--box_pose_episode_hdf5",
        type=str,
        default=None,
        help="If set, uses HDF5 attr env_state0 as BOX_POSE for replay.",
    )
    args = ap.parse_args()

    if args.box_pose_episode_hdf5:
        import h5py

        with h5py.File(args.box_pose_episode_hdf5, "r") as root:
            if "env_state0" not in root.attrs:
                raise SystemExit("box_pose_episode_hdf5 missing attr env_state0")
            BOX_POSE[0] = np.array(root.attrs["env_state0"], dtype=np.float32)

    with open(args.quantizer_json, "r", encoding="utf-8") as f:
        qj = json.load(f)
    q = ActionQuantizer(
        mins=np.asarray(qj["mins"], dtype=np.float32),
        maxs=np.asarray(qj["maxs"], dtype=np.float32),
        bins=args.bins,
    )

    raw = extract_text_from_ollama_response(args.ollama_response_json)
    act_block = extract_act_block(raw)
    if not act_block:
        raise SystemExit("Could not find ACT_START/ACT_END in response.")

    codes, errors = parse_action_block(act_block, M_expected=args.M)
    os.makedirs("artifacts/parsed_actions", exist_ok=True)
    parsed_path = os.path.join(
        "artifacts/parsed_actions",
        os.path.basename(args.ollama_response_json).replace(".json", ".npz"),
    )
    if codes is None:
        np.savez(parsed_path, ok=False, errors=np.array(errors, dtype=object))
        raise SystemExit(f"Failed to parse action block. Errors: {errors}")

    # Map gripper bits (0/1) to quantizer bin extremes.
    codes = codes.copy()
    codes[:, 6] = codes[:, 6] * (args.bins - 1)
    codes[:, 13] = codes[:, 13] * (args.bins - 1)

    actions_wp = q.decode(codes).astype(np.float32)
    actions = upsample_waypoints_piecewise_constant(actions_wp, T=args.episode_len)
    np.savez(parsed_path, ok=True, errors=np.array(errors, dtype=object), codes=codes, actions=actions)

    res = replay_actions(actions, episode_len=args.episode_len, video_path=args.video_path)
    print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()

