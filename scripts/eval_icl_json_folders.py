import argparse
import csv
import json
import os
import sys
import time

import h5py

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.episodes import episode_path, list_episode_ids, read_env_state0, read_top_frame
from act_kat.icl import (
    actions_json_to_codes,
    anchor_descriptors_from_episode,
    build_demo_pairs,
    codes_to_trajectory,
    fit_quantizer_from_episodes,
    generate_icl_actions,
    parse_icl_actions_json,
    tokenize_obs,
)
from sim_env import BOX_POSE


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ICL (JSON actions) on demo/test folder splits.")
    ap.add_argument("--demos_dir", type=str, required=True)
    ap.add_argument("--tests_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemma4:26b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--obs_bins", type=int, default=16)
    ap.add_argument(
        "--keypoint_mode",
        type=str,
        default="anchored",
        choices=["anchored", "fps"],
        help="anchored: paper-style consistent KP ids; fps: legacy per-frame FPS",
    )
    ap.add_argument(
        "--token_format",
        type=str,
        default="keypoints",
        choices=["keypoints", "legacy"],
        help="keypoints: DINO anchored KP i x y d; legacy: optional fps keypoints",
    )
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_csv", type=str, default="artifacts/results_icl_json.csv")
    ap.add_argument("--max_tests", type=int, default=None)
    ap.add_argument(
        "--upsample",
        type=str,
        default="linear",
        choices=["linear", "hold"],
        help="Waypoint expansion for rollout (default: linear)",
    )
    args = ap.parse_args()

    demo_ids = list_episode_ids(args.demos_dir)
    test_ids = list_episode_ids(args.tests_dir)
    if args.max_tests is not None:
        test_ids = test_ids[: int(args.max_tests)]
    if not demo_ids:
        raise SystemExit("No demos found")
    if not test_ids:
        raise SystemExit("No tests found")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs("artifacts/replays", exist_ok=True)

    demo_paths = [episode_path(args.demos_dir, ep) for ep in demo_ids]
    quant = fit_quantizer_from_episodes(demo_paths, bins=args.bins)
    anchor_desc = None
    if args.keypoint_mode == "anchored":
        anchor_desc = anchor_descriptors_from_episode(demo_paths[0], args.K, device=args.device)
        print(f"  [anchors] {anchor_desc.shape} descriptors from {demo_paths[0]}")
    demo_pairs = build_demo_pairs(
        demo_paths,
        quant,
        K=args.K,
        M=args.M,
        device=args.device,
        obs_bins=args.obs_bins,
        anchor_desc=anchor_desc,
        keypoint_mode=args.keypoint_mode,
        token_format=args.token_format,
    )

    from act_kat.replay import replay_actions

    rows = []
    successes = 0
    parse_errors = 0
    gen_times = []

    for idx, ep in enumerate(test_ids):
        test_path = episode_path(args.tests_dir, ep)
        with h5py.File(test_path, "r") as root:
            img, depth = read_top_frame(root, t=0)
            env0 = read_env_state0(root)
        query_obs = tokenize_obs(
            img,
            depth=depth,
            k=args.K,
            device=args.device,
            obs_bins=args.obs_bins,
            anchor_desc=anchor_desc,
            keypoint_mode=args.keypoint_mode,
            token_format=args.token_format,
        )

        t0 = time.time()
        run_id = f"eval_icl_json_demo{len(demo_ids)}_test{ep}_idx{idx}"
        text = generate_icl_actions(
            demo_pairs=demo_pairs,
            query_obs=query_obs,
            M=args.M,
            model=args.model,
            ollama_url=args.ollama_url,
            run_id=run_id,
        )
        gen_times.append(time.time() - t0)
        print(f"  [saved] artifacts/responses/{run_id}.json")

        try:
            actions = parse_icl_actions_json(text, args.M)
            codes = actions_json_to_codes(actions, bins=args.bins)
        except Exception as e:
            parse_errors += 1
            rows.append(
                {
                    "episode_id": ep,
                    "success": 0,
                    "max_reward": 0.0,
                    "parse_ok": 0,
                    "parse_errors": json.dumps([f"json_parse:{type(e).__name__}"]),
                }
            )
            continue

        trajectory = codes_to_trajectory(
            codes, quant, episode_len=args.episode_len, upsample=args.upsample
        )
        BOX_POSE[0] = env0
        res = replay_actions(
            trajectory,
            episode_len=args.episode_len,
            video_path=f"artifacts/replays/eval_icl_json_demo{len(demo_ids)}_test{ep}_idx{idx}.mp4",
        )

        successes += int(res.success)
        rows.append(
            {
                "episode_id": ep,
                "success": int(res.success),
                "max_reward": res.max_reward,
                "parse_ok": 1,
                "parse_errors": "[]",
            }
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["episode_id", "success", "max_reward", "parse_ok", "parse_errors"],
        )
        w.writeheader()
        w.writerows(rows)

    summary = {
        "demos_dir": args.demos_dir,
        "tests_dir": args.tests_dir,
        "n_demos": len(demo_ids),
        "n_tests": len(test_ids),
        "K_keypoints": args.K,
        "M_actions": args.M,
        "success_rate": successes / max(1, len(test_ids)),
        "parse_error_rate": parse_errors / max(1, len(test_ids)),
        "avg_gen_time_s": float(sum(gen_times) / len(gen_times)) if gen_times else None,
        "out_csv": args.out_csv,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
