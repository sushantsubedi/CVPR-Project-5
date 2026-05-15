#!/usr/bin/env python3
"""Download DINO ViT-S/16 weights to assets/weights/ (Facebook CDN, no Hugging Face)."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from act_kat.vision_tokens import DINO_CHECKPOINT_PATH, DINO_WEIGHTS_URL

DEFAULT_OUT = DINO_CHECKPOINT_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        print(f"Already exists: {out} ({out.stat().st_size} bytes)")
        return

    print(f"Downloading {DINO_WEIGHTS_URL}")
    print(f"  -> {out}")
    urllib.request.urlretrieve(DINO_WEIGHTS_URL, out)
    print("Done.")


if __name__ == "__main__":
    main()
