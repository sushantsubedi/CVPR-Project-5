import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.episodes import episode_path, read_top_frame
from act_kat.ollama_client import prompt_txt_path, response_json_path
from act_kat.icl import (
    anchor_descriptors_from_episode,
    build_demo_pairs,
    fit_quantizer_from_episodes,
    generate_icl_actions,
    tokenize_obs,
)
import h5py


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one ICL query via Ollama (JSON-constrained actions).")
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--demo_ids", type=str, required=True, help="Comma-separated demo episode ids, e.g. 0,1")
    ap.add_argument("--query_id", type=int, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--obs_bins", type=int, default=64)
    ap.add_argument("--model", type=str, default="gemma4:26b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument(
        "--token_format",
        type=str,
        default="keypoints",
        choices=["keypoints", "legacy"],
    )
    args = ap.parse_args()

    demo_ids = [int(x) for x in args.demo_ids.split(",") if x.strip()]
    demo_paths = [episode_path(args.dataset_dir, i) for i in demo_ids]
    query_path = episode_path(args.dataset_dir, args.query_id)

    quant = fit_quantizer_from_episodes(demo_paths + [query_path], bins=args.bins)
    anchor_desc = anchor_descriptors_from_episode(demo_paths[0], args.K, device=args.device)
    demo_pairs = build_demo_pairs(
        demo_paths,
        quant,
        K=args.K,
        M=args.M,
        device=args.device,
        obs_bins=args.obs_bins,
        anchor_desc=anchor_desc,
        keypoint_mode="anchored",
        token_format=args.token_format,
    )

    with h5py.File(query_path, "r") as root:
        img, depth = read_top_frame(root, t=0)
    query_obs = tokenize_obs(
        img,
        depth=depth,
        k=args.K,
        device=args.device,
        obs_bins=args.obs_bins,
        anchor_desc=anchor_desc,
        keypoint_mode="anchored",
        token_format=args.token_format,
    )

    run_id = args.run_id or "run"
    text = generate_icl_actions(
        demo_pairs=demo_pairs,
        query_obs=query_obs,
        M=args.M,
        model=args.model,
        ollama_url=args.ollama_url,
        run_id=run_id,
    )
    print(text)
    print(f"[saved_prompt] {prompt_txt_path('artifacts', run_id)}")
    print(f"[saved_response] {response_json_path('artifacts', run_id)}")


if __name__ == "__main__":
    main()
