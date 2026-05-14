import argparse
import os
import sys

# Allow running from repo root without installation.
sys.path.insert(0, os.path.abspath(os.getcwd()))

from act_kat.episode_schema import write_schema_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="artifacts/schema_report.jsonl")
    args = ap.parse_args()
    write_schema_report(args.dataset_dir, args.out)


if __name__ == "__main__":
    main()

