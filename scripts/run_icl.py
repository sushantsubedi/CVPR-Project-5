import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.action_tokens import ActionQuantizer, action_tokens_from_episode
from act_kat.ollama_client import ollama_chat, ollama_generate
from act_kat.prompting import SYSTEM_INSTRUCTION, build_chat_user_message
from act_kat.vision_tokens import tokenize_keypoints_2d


def load_obs_action(episode_hdf5: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(episode_hdf5, "r") as root:
        img0 = root["/observations/images/top"][0]
        act = root["/action"][()].astype(np.float32)
    return img0, act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--demo_ids", type=str, required=True, help="Comma-separated demo episode ids, e.g. 0,1")
    ap.add_argument("--query_id", type=int, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--model", type=str, default="gemma4:26b")
    ap.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument(
        "--mode",
        type=str,
        default="icl",
        choices=["icl", "icl_json", "echo_gt"],
        help="icl: few-shot prompt; icl_json: few-shot but force JSON output; echo_gt: repeat ground-truth ACT block (sanity).",
    )
    args = ap.parse_args()

    demo_ids = [int(x) for x in args.demo_ids.split(",") if x.strip() != ""]
    demo_eps = [os.path.join(args.dataset_dir, f"episode_{i}.hdf5") for i in demo_ids]
    query_ep = os.path.join(args.dataset_dir, f"episode_{args.query_id}.hdf5")

    # Build quantizer from demos+query (keeps decode stable for this run)
    all_act = []
    for p in demo_eps + [query_ep]:
        _, act = load_obs_action(p)
        all_act.append(act)
    all_act = np.concatenate(all_act, axis=0)
    q = ActionQuantizer(mins=all_act.min(axis=0), maxs=all_act.max(axis=0), bins=args.bins)

    demo_pairs = []
    for p in demo_eps:
        img0, act = load_obs_action(p)
        _, obs_tok = tokenize_keypoints_2d(img0, k=args.K, device=args.device)
        _, act_tok = action_tokens_from_episode(act, quantizer=q, M=args.M)
        demo_pairs.append(obs_tok + act_tok)

    img0q, _actq = load_obs_action(query_ep)
    _, query_obs = tokenize_keypoints_2d(img0q, k=args.K, device=args.device)

    if args.mode == "icl":
        user_msg = build_chat_user_message(demo_pairs, query_obs=query_obs)
        system_msg = SYSTEM_INSTRUCTION
        format_arg = None
    elif args.mode == "icl_json":
        # Force JSON output and use /api/generate to avoid chat reasoning-only responses.
        prompt = build_chat_user_message(demo_pairs, query_obs=query_obs) + (
            "\n\nReturn ONLY JSON matching the provided schema.\n"
            f"Output exactly M={args.M} action objects.\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": args.M,
                    "maxItems": args.M,
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

        resp = ollama_generate(
            prompt=prompt,
            model=args.model,
            url=args.ollama_url,
            options={"temperature": 0.0, "num_predict": 2048},
            run_id=args.run_id,
            artifacts_dir="artifacts",
            format=schema,
        )
        text = resp.get("response", "") if isinstance(resp, dict) else ""
        print(text)
        out_dir = os.path.join("artifacts", "icl_json_outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.run_id or 'run'}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[saved_json] {out_path}")
        return
    else:
        # Sanity mode: ask the model to repeat the ground-truth ACT token block
        _, act_tok_gt = action_tokens_from_episode(_actq, quantizer=q, M=args.M)
        system_msg = (
            "Output EXACTLY the provided ACT block, from ACT_START to ACT_END. "
            "No other text."
        )
        user_msg = "Repeat this ACT block exactly:\n\n" + act_tok_gt
        format_arg = None

    options = {"temperature": 0.2, "top_p": 0.9, "num_predict": 2048}

    resp = ollama_chat(
        system=system_msg,
        user=user_msg,
        model=args.model,
        url=args.ollama_url,
        options=options,
        run_id=args.run_id,
        artifacts_dir="artifacts",
        format=format_arg,
    )

    msg = resp.get("message", {}) if isinstance(resp, dict) else {}
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    thinking = msg.get("thinking", "") if isinstance(msg, dict) else ""
    # Print best-effort raw output for convenience.
    print(content if content else thinking)

    # icl_json returns earlier


if __name__ == "__main__":
    main()

