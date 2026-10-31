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

  # Matched control: 2B Qwen (--laptop) at the LoRA test resolution, into its own folder
  python scripts/run_protocol_fair_eval.py --phase qwen-leakfree --laptop --max-side 384 \
      --results-subdir results/protocol/random_evidence_only_384 --tag _384

Outputs never overwrite an existing file (a timestamped name is used instead).
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
    p.add_argument("--max-side", type=int, default=None, help="longest image side in px (default: 336 with --laptop, else native)")
    p.add_argument("--results-subdir", default="results/protocol")
    p.add_argument("--tag", default="", help="suffix for output file names, e.g. _384")
    p.add_argument("--combo", action="append", dest="combos", default=None, help="restrict to these combo ids")
    p.add_argument("--text-mode", default="compact", choices=("compact", "full", "evidence_only"),
                   help="fair-four phase only: text input (default compact; 'full' contains the label line)")
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
            combo_ids=args.combos or ["qwen_vl_custom_mlp_qwen"],
            laptop=args.laptop,
            purge_after=args.purge_after,
            splits_dir=splits,
            text_mode="evidence_only",
            results_subdir=args.results_subdir,
            max_side=args.max_side,
            tag=args.tag,
        )
        print(f"wrote {args.results_subdir}/vlm_combo_comparison_evidence_only{args.tag}.json")
        return

    if args.phase == "fair-four":
        run_all(
            ROOT,
            split=args.split,
            max_eval=args.max_eval,
            dry_run=args.dry_run,
            combo_ids=args.combos or [
                "qwen_vl_custom_mlp_qwen",
                "llava15_clip_mlp_vicuna",
                "paligemma_siglip_linear_gemma",
                "blip2_clip_qformer_opt",
            ],
            laptop=args.laptop,
            purge_after=args.purge_after,
            splits_dir=splits,
            text_mode=args.text_mode,
            results_subdir=args.results_subdir,
            max_side=args.max_side,
            tag=args.tag,
        )
        print(f"wrote {args.results_subdir}/ (text_mode={args.text_mode}, tag={args.tag})")
        return


if __name__ == "__main__":
    main()
