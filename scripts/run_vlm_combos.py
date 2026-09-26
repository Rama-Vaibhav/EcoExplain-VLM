#!/usr/bin/env python3
"""Evaluate VLM combos on the same EcoExplain split.

  python scripts/run_vlm_combos.py --dry-run --max-eval 8
  python scripts/run_vlm_combos.py --max-eval 8
  python scripts/run_vlm_combos.py --max-eval 55 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm_eval import run_all  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--max-eval", type=int, default=55)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--laptop",
        action="store_true",
        help="Use 0.5B–3B checkpoints that fit a 16 GB Mac; do not load 7B weights.",
    )
    p.add_argument(
        "--purge-after",
        action="store_true",
        help="Delete each model's local HF cache after scoring so disk stays free.",
    )
    p.add_argument(
        "--combos",
        nargs="*",
        default=[
            "llava15_clip_mlp_vicuna",
            "paligemma_siglip_linear_gemma",
            "qwen_vl_custom_mlp_qwen",
            "blip2_clip_qformer_opt",
        ],
    )
    args = p.parse_args()
    comparison = run_all(
        ROOT,
        split=args.split,
        max_eval=args.max_eval,
        dry_run=args.dry_run,
        combo_ids=args.combos,
        laptop=args.laptop,
        purge_after=args.purge_after,
    )
    print("best", comparison.get("best_combo_id"), comparison.get("best_accuracy"))
    print("wrote results/vlm_combo_comparison.json")


if __name__ == "__main__":
    main()
