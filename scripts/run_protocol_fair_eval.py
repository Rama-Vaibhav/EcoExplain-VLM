#!/usr/bin/env python3
"""Re-run evaluations for protocol fixes (leak-free, fair compact text, spatial split).

Examples (repo root, venv active):

  # 1) Spatial split files only (no GPU)
  python scripts/write_spatial_splits.py

  # 2) Qwen leak-free zero-shot on all 55 test patches (laptop; ~30–90 min)
  python scripts/run_protocol_fair_eval.py --phase qwen-leakfree

  # 3) All four models, same compact evidence text (laptop; hours if run sequentially)
  python scripts/run_protocol_fair_eval.py --phase fair-four --laptop --purge-after

  # 4) Same as (3) on spatial test set
  python scripts/run_protocol_fair_eval.py --phase fair-four --splits-dir data/splits_spatial
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
    p.add_argument(
        "--phase",
        choices=("qwen-leakfree", "fair-four", "spatial-manifest-only"),
        required=True,
    )
    p.add_argument("--max-eval", type=int, default=55)
    p.add_argument("--split", default="test")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--laptop", action="store_true")
    p.add_argument("--purge-after", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.phase == "spatial-manifest-only":
        from src.split_utils import write_spatial_block_splits

        write_spatial_block_splits(ROOT)
        return

    splits = ROOT / args.splits_dir
    if args.phase == "qwen-leakfree":
        run_all(
            ROOT,
            split=args.split,
            max_eval=args.max_eval,
            dry_run=args.dry_run,
            combo_ids=["qwen_vl_custom_mlp_qwen"],
            laptop=args.laptop,
            purge_after=args.purge_after,
            splits_dir=splits,
            text_mode="evidence_only",
            results_subdir="results/protocol",
        )
        print("wrote results/protocol/vlm_combo_comparison_evidence_only.json")
        return

    if args.phase == "fair-four":
        run_all(
            ROOT,
            split=args.split,
            max_eval=args.max_eval,
            dry_run=args.dry_run,
            combo_ids=[
                "qwen_vl_custom_mlp_qwen",
                "llava15_clip_mlp_vicuna",
                "paligemma_siglip_linear_gemma",
                "blip2_clip_qformer_opt",
            ],
            laptop=args.laptop,
            purge_after=args.purge_after,
            splits_dir=splits,
            text_mode="compact",
            results_subdir="results/protocol",
        )
        print("wrote results/protocol/vlm_combo_comparison_compact.json")
        return


if __name__ == "__main__":
    main()
