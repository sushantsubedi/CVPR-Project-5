import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer
from act_kat.replay import replay_actions, upsample_waypoints_piecewise_constant
from sim_env import BOX_POSE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--icl_json", type=str, required=True, help="Path to JSON with {'actions':[...]} output by LLM")
    ap.add_argument("--quantizer_json", type=str, required=True)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--video_path", type=str, default="artifacts/replays/replay_from_icl_json.mp4")
    ap.add_argument("--box_pose_episode_hdf5", type=str, required=True)
    args = ap.parse_args()

    with open(args.quantizer_json, "r", encoding="utf-8") as f:
        qj = json.load(f)
    quant = ActionQuantizer(
        mins=np.asarray(qj["mins"], dtype=np.float32),
        maxs=np.asarray(qj["maxs"], dtype=np.float32),
        bins=args.bins,
    )

    with open(args.icl_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) == 0:
        raise SystemExit("Invalid JSON: expected non-empty 'actions' list")

    codes = np.zeros((len(actions), 14), dtype=np.int32)
    for i, a in enumerate(actions):
        L = a.get("L"); R = a.get("R"); LG = a.get("LG"); RG = a.get("RG")
        if not (isinstance(L, list) and len(L) == 6 and isinstance(R, list) and len(R) == 6):
            raise SystemExit(f"Bad L/R at i={i}")
        codes[i, :6] = np.asarray([int(x) for x in L], dtype=np.int32)
        codes[i, 7:13] = np.asarray([int(x) for x in R], dtype=np.int32)
        codes[i, 6] = int(LG) * (args.bins - 1)
        codes[i, 13] = int(RG) * (args.bins - 1)

    # Hold/repeat: spread M waypoints over the full episode length.
    actions_wp = quant.decode(codes).astype(np.float32)
    actions_full = upsample_waypoints_piecewise_constant(actions_wp, T=args.episode_len)

    with h5py.File(args.box_pose_episode_hdf5, "r") as root:
        BOX_POSE[0] = np.array(root.attrs["env_state0"], dtype=np.float32)

    res = replay_actions(actions_full, episode_len=args.episode_len, video_path=args.video_path)
    print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()

