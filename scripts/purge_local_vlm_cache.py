#!/usr/bin/env python3
"""Delete local Hugging Face VLM shards. The official weights stay on the Hub.

  python scripts/purge_local_vlm_cache.py
  python scripts/purge_local_vlm_cache.py --repos llava-hf/llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm_eval import hf_hub_dir, purge_hf_repo  # noqa: E402

DEFAULT_REPOS = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-interleave-qwen-0.5b-hf",
    "google/paligemma-3b-mix-224",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Salesforce/blip2-opt-2.7b",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for repo in args.repos:
        d = hf_hub_dir(repo)
        if not d.exists():
            print("absent", repo)
            continue
        print("found", repo, "->", d)
        if not args.dry_run:
            purge_hf_repo(repo)


if __name__ == "__main__":
    main()
