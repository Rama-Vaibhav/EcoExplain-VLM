#!/usr/bin/env python3
"""Closed-set eval for LLaVA 0.5B, PaliGemma 3B, and BLIP-2 OPT 2.7B.

  python scripts/run_vlm_closedset.py --dry-run
  python scripts/run_vlm_closedset.py --max-eval 4
  python scripts/run_vlm_closedset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm_closedset import CLOSEDSET_COMBOS, answer_prompt, run_closedset  # noqa: E402
from src.vlm_combos import load_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-eval", type=int, default=55)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-laptop",
        action="store_true",
        help="Load LLaVA-1.5 7B instead of llava-interleave 0.5B. PaliGemma and BLIP-2 ids stay the same.",
    )
    p.add_argument("--combos", nargs="*", default=list(CLOSEDSET_COMBOS))
    args = p.parse_args()

    if args.dry_run:
        rec = load_jsonl(ROOT / "dataset.jsonl")[0]
        text = answer_prompt(rec)
        print(text)
        if "candidate_class" in text:
            raise SystemExit("dry-run failed: prompt contains candidate_class")
        print("dry-run ok", "id", rec["id"], "gold", rec.get("candidate_class"))
        return

    run_closedset(ROOT, args.combos, laptop=not args.no_laptop, max_eval=args.max_eval)


if __name__ == "__main__":
    main()
