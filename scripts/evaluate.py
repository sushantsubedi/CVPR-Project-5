"""KAT-ICL evaluation, ablations, and aggregation in one script.

Modes
-----
Single run (one (demos, tests, K, M) configuration):
  python3 scripts/evaluate.py run \
      --demos_dir data/transfer_cube/subsets/demos_5 \
      --tests_dir data/transfer_cube/subsets/test_5 \
      --eval_split both --K 10 --M 20

Full sweep (seen/unseen on demos_5 + n_demos / K / M / model ablations + aggregate):
  python3 scripts/evaluate.py sweep
      [--collection_seed 0] [--model gemma4:26b]
      [--models 'gemma4:26b,llama3.2:latest'] [--max_tests 5]

Aggregate existing CSVs only (no LLM calls):
  python3 scripts/evaluate.py aggregate --glob 'artifacts/results_*.csv'

Splits
------
  seen   = leave-one-out queries on the demo folder (query in demos, prompt holds rest)
  unseen = queries on a separate held-out folder (test_5)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from act_kat.episodes import (
    episode_path,
    list_episode_ids,
    read_env_state0,
    read_top_frame,
)
from act_kat.icl import (
    actions_to_codes,
    anchor_descriptors_from_episode,
    build_demo_pairs,
    codes_to_trajectory,
    fit_quantizer_from_episodes,
    generate_icl_actions,
    parse_actions_strict,
    tokenize_obs,
)
from act_kat.replay import replay_actions
from sim_env import BOX_POSE


# ---------------------------------------------------------------------------
# Single run (seen / unseen / both)
# ---------------------------------------------------------------------------

def _default_results_csv(demos_dir: str, tests_dir: str, split: str) -> str:
    a = os.path.basename(os.path.normpath(demos_dir))
    b = os.path.basename(os.path.normpath(tests_dir))
    return f"artifacts/results_{a}_{b}_{split}.csv"


def _query_plan(
    eval_split: str, demo_ids: List[int], test_ids: List[int]
) -> List[Tuple[str, int, List[int]]]:
    """Return list of (split_label, query_episode_id, prompt_demo_ids)."""
    if eval_split == "unseen":
        return [("unseen", ep, demo_ids) for ep in test_ids]
    if eval_split == "seen":
        plan: List[Tuple[str, int, List[int]]] = []
        for q in demo_ids:
            others = [d for d in demo_ids if d != q]
            if others:
                plan.append(("seen", q, others))
        return plan
    if eval_split == "both":
        return (
            _query_plan("seen", demo_ids, test_ids)
            + _query_plan("unseen", demo_ids, test_ids)
        )
    raise ValueError(f"unknown eval_split {eval_split!r}")


def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    """Run ICL on (demos, tests) and write a results CSV + summary JSON."""
    out_csv = args.out_csv or _default_results_csv(
        args.demos_dir, args.tests_dir, args.eval_split
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    for sub in ("replays", "prompts", "responses"):
        os.makedirs(f"artifacts/{sub}", exist_ok=True)

    demo_ids = list_episode_ids(args.demos_dir)
    test_ids = list_episode_ids(args.tests_dir)
    if args.max_tests is not None:
        test_ids = test_ids[: int(args.max_tests)]
    if not demo_ids:
        raise SystemExit(f"No demos in {args.demos_dir}")
    if args.eval_split in ("unseen", "both") and not test_ids:
        raise SystemExit(f"No tests in {args.tests_dir}")

    demo_paths_all = [episode_path(args.demos_dir, ep) for ep in demo_ids]
    quant = fit_quantizer_from_episodes(demo_paths_all, bins=args.bins)
    anchor_desc = anchor_descriptors_from_episode(
        demo_paths_all[0], args.K, device=args.device
    )
    print(f"[results] {out_csv}  split={args.eval_split}  K={args.K} M={args.M}")
    print(f"  [anchors] {anchor_desc.shape} from {demo_paths_all[0]}")

    plan = _query_plan(args.eval_split, demo_ids, test_ids)
    rows: List[Dict[str, Any]] = []
    gen_times: List[float] = []
    successes = 0
    parse_errors = 0

    for idx, (split_label, query_ep, prompt_ids) in enumerate(plan):
        prompt_paths = [episode_path(args.demos_dir, d) for d in prompt_ids]
        demo_pairs = build_demo_pairs(
            prompt_paths, quant,
            K=args.K, M=args.M,
            anchor_desc=anchor_desc, device=args.device, obs_bins=args.obs_bins,
        )

        query_dir = args.demos_dir if split_label == "seen" else args.tests_dir
        test_path = episode_path(query_dir, query_ep)
        with h5py.File(test_path, "r") as root:
            img, depth = read_top_frame(root, t=0)
            env0 = read_env_state0(root)

        query_obs = tokenize_obs(
            img, anchor_desc=anchor_desc, depth=depth,
            device=args.device, obs_bins=args.obs_bins,
        )

        run_id = f"eval_{split_label}_demo{len(prompt_ids)}_q{query_ep}_idx{idx}"
        t0 = time.time()
        text = generate_icl_actions(
            demo_pairs=demo_pairs, query_obs=query_obs, M=args.M,
            model=args.model, ollama_url=args.ollama_url, run_id=run_id,
        )
        gen_times.append(time.time() - t0)
        print(f"  [{split_label}] ep={query_ep} demos={prompt_ids} -> {run_id}.json")

        row_base = {
            "split": split_label,
            "episode_id": query_ep,
            "model": args.model,
            "K": args.K,
            "M": args.M,
            "n_demos_in_prompt": len(prompt_ids),
            "demo_ids_in_prompt": json.dumps(prompt_ids),
        }
        try:
            actions = parse_actions_strict(text, args.M)
            codes = actions_to_codes(actions, bins=args.bins)
        except Exception as e:
            parse_errors += 1
            rows.append({
                **row_base, "success": 0, "max_reward": 0.0, "parse_ok": 0,
                "parse_errors": json.dumps([f"json_parse:{type(e).__name__}"]),
            })
            continue

        trajectory = codes_to_trajectory(
            codes, quant, episode_len=args.episode_len, upsample=args.upsample
        )
        BOX_POSE[0] = env0
        res = replay_actions(
            trajectory, episode_len=args.episode_len,
            video_path=f"artifacts/replays/{run_id}.mp4",
        )
        successes += int(res.success)
        rows.append({
            **row_base, "success": int(res.success),
            "max_reward": res.max_reward, "parse_ok": 1, "parse_errors": "[]",
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split", "episode_id", "model", "K", "M",
                "n_demos_in_prompt", "demo_ids_in_prompt",
                "success", "max_reward", "parse_ok", "parse_errors",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    seen_rows = [r for r in rows if r["split"] == "seen"]
    unseen_rows = [r for r in rows if r["split"] == "unseen"]

    def _rate(sub: List[Dict[str, Any]]) -> Optional[float]:
        return sum(int(r["success"]) for r in sub) / len(sub) if sub else None

    summary = {
        "demos_dir": args.demos_dir,
        "tests_dir": args.tests_dir,
        "eval_split": args.eval_split,
        "model": args.model,
        "collection_seed": args.collection_seed,
        "K_keypoints": args.K,
        "M_actions": args.M,
        "bins": args.bins,
        "obs_bins": args.obs_bins,
        "n_queries": n,
        "success_rate_all": successes / max(1, n),
        "success_rate_seen": _rate(seen_rows),
        "success_rate_unseen": _rate(unseen_rows),
        "parse_error_rate": parse_errors / max(1, n),
        "avg_gen_time_s": (sum(gen_times) / len(gen_times)) if gen_times else None,
        "out_csv": out_csv,
    }
    summary_path = os.path.splitext(out_csv)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[summary] {summary_path}")
    return summary


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--demos_dir", required=True)
    ap.add_argument("--tests_dir", required=True)
    ap.add_argument(
        "--eval_split", default="both", choices=["seen", "unseen", "both"],
        help="seen=LOTO on demos; unseen=held-out tests; both=run both and tag rows",
    )
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--obs_bins", type=int, default=16)
    ap.add_argument("--episode_len", type=int, default=400)
    ap.add_argument("--upsample", choices=["linear", "hold"], default="linear")
    ap.add_argument("--max_tests", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma4:26b"))
    ap.add_argument("--ollama_url", default="http://127.0.0.1:11434")
    ap.add_argument("--collection_seed", type=int, default=0,
                    help="dataset RNG seed (metadata only; set when recording)")


# ---------------------------------------------------------------------------
# Full sweep (Step 5)
# ---------------------------------------------------------------------------

_SUB = ROOT / "data" / "transfer_cube" / "subsets"


def _model_tag(model: str) -> str:
    """Filename-safe slug for an Ollama model name (e.g. 'llama3.2:latest' -> 'llama3_2_latest')."""
    return "".join(c if c.isalnum() else "_" for c in model).strip("_")


def _sweep_one(
    *,
    demos_dir: str,
    tests_dir: str,
    eval_split: str,
    out_csv: str,
    K: int = 10,
    M: int = 20,
    model: Optional[str] = None,
    args: argparse.Namespace,
) -> None:
    ns = argparse.Namespace(
        demos_dir=demos_dir, tests_dir=tests_dir, eval_split=eval_split,
        out_csv=out_csv, K=K, M=M,
        bins=args.bins, obs_bins=args.obs_bins,
        episode_len=args.episode_len, upsample=args.upsample,
        max_tests=args.max_tests, device=args.device,
        model=model or args.model, ollama_url=args.ollama_url,
        collection_seed=args.collection_seed,
    )
    run_one(ns)


def run_sweep(args: argparse.Namespace) -> None:
    """Run the full evaluation sweep (Step 5) then aggregate."""
    tests = str(_SUB / "test_5")

    # (a) seen + unseen on demos_5 with K=10, M=20 — main result
    _sweep_one(
        demos_dir=str(_SUB / "demos_5"), tests_dir=tests,
        eval_split="both", out_csv="artifacts/results_main_seen_unseen.csv",
        args=args,
    )

    # (b) #demos ablation (unseen only, K=10, M=20).
    # demos_5 is the main config (covered by (a) on the unseen rows), so the
    # ablation only sweeps the additional 10 / 20 configurations here.
    for n, folder in [(10, "demos_10"), (20, "demos_20")]:
        _sweep_one(
            demos_dir=str(_SUB / folder), tests_dir=tests,
            eval_split="unseen", out_csv=f"artifacts/results_ablation_ndemos_{n}_unseen.csv",
            args=args,
        )

    # (c) K ablation (demos_5, unseen)
    for k in (5, 10, 20):
        _sweep_one(
            demos_dir=str(_SUB / "demos_5"), tests_dir=tests,
            eval_split="unseen", out_csv=f"artifacts/results_ablation_K{k}_unseen.csv",
            K=k, args=args,
        )

    # (d) M ablation (demos_5, unseen)
    for m in (10, 20, 40):
        _sweep_one(
            demos_dir=str(_SUB / "demos_5"), tests_dir=tests,
            eval_split="unseen", out_csv=f"artifacts/results_ablation_M{m}_unseen.csv",
            M=m, args=args,
        )

    # (e) Model ablation (demos_5, K=10, M=20, unseen) across Ollama models.
    for model in args.models:
        _sweep_one(
            demos_dir=str(_SUB / "demos_5"), tests_dir=tests,
            eval_split="unseen",
            out_csv=f"artifacts/results_ablation_model_{_model_tag(model)}_unseen.csv",
            model=model, args=args,
        )

    aggregate(argparse.Namespace(
        glob="artifacts/results_*.csv",
        out_json="artifacts/evaluation_summary.json",
        out_md="artifacts/evaluation_tables.md",
    ))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _rate(sub: List[Dict[str, Any]]) -> float:
    if not sub:
        return float("nan")
    return sum(int(r.get("success", 0)) for r in sub) / len(sub)


def summarize_csv(path: str) -> Dict[str, Any]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"path": path, "n": 0}
    seen = [r for r in rows if r.get("split") == "seen"]
    unseen = [r for r in rows if r.get("split") == "unseen"]
    out: Dict[str, Any] = {
        "path": path,
        "n": len(rows),
        "parse_ok_rate": sum(int(r.get("parse_ok", 0)) for r in rows) / len(rows),
    }
    if seen or unseen:
        out["success_rate_seen"] = _rate(seen)
        out["success_rate_unseen"] = _rate(unseen)
        out["n_seen"] = len(seen)
        out["n_unseen"] = len(unseen)
    else:
        out["success_rate"] = _rate(rows)
    return out


def _fmt_pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{100 * float(x):.1f}%"


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files match {args.glob}")

    summaries = [summarize_csv(p) for p in paths]
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"files": summaries}, f, indent=2)

    lines = [
        "# P5 Step 5 — aggregated results",
        "",
        "| CSV | N | Seen SR | Unseen SR | Overall SR | Parse OK |",
        "|-----|--:|--------:|----------:|-----------:|---------:|",
    ]
    for s in summaries:
        lines.append(
            "| `{name}` | {n} | {seen} | {unseen} | {overall} | {parse} |".format(
                name=os.path.basename(s["path"]),
                n=s["n"],
                seen=_fmt_pct(s.get("success_rate_seen")),
                unseen=_fmt_pct(s.get("success_rate_unseen")),
                overall=_fmt_pct(s.get("success_rate")),
                parse=_fmt_pct(s.get("parse_ok_rate")),
            )
        )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out_json} and {args.out_md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    top = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run", help="Run one (demos, tests, K, M) configuration")
    _add_run_args(ap_run)

    ap_sweep = sub.add_parser("sweep", help="Run full P5 Step 5 sweep and aggregate")
    ap_sweep.add_argument("--K", type=int, default=10)
    ap_sweep.add_argument("--M", type=int, default=20)
    ap_sweep.add_argument("--bins", type=int, default=32)
    ap_sweep.add_argument("--obs_bins", type=int, default=16)
    ap_sweep.add_argument("--episode_len", type=int, default=400)
    ap_sweep.add_argument("--upsample", choices=["linear", "hold"], default="linear")
    ap_sweep.add_argument("--max_tests", type=int, default=None)
    ap_sweep.add_argument("--device", default="cpu")
    ap_sweep.add_argument(
        "--model", default=os.environ.get("OLLAMA_MODEL", "gemma4:26b"),
        help="Default Ollama model used by (a)-(d). Model ablation uses --models.",
    )
    ap_sweep.add_argument(
        "--models",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=["gemma4:26b", "llama3.2:latest"],
        help="Comma-separated Ollama models for the (e) model ablation pass.",
    )
    ap_sweep.add_argument("--ollama_url", default="http://127.0.0.1:11434")
    ap_sweep.add_argument("--collection_seed", type=int, default=0)

    ap_agg = sub.add_parser("aggregate", help="Aggregate result CSVs into summary tables")
    ap_agg.add_argument("--glob", default="artifacts/results_*.csv")
    ap_agg.add_argument("--out_json", default="artifacts/evaluation_summary.json")
    ap_agg.add_argument("--out_md", default="artifacts/evaluation_tables.md")

    args = top.parse_args()
    if args.cmd == "run":
        run_one(args)
    elif args.cmd == "sweep":
        run_sweep(args)
    elif args.cmd == "aggregate":
        aggregate(args)


if __name__ == "__main__":
    main()
