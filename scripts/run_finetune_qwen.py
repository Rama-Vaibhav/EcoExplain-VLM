#!/usr/bin/env python3
"""LoRA fine-tune Qwen2-VL on EcoExplain train split; eval on test.

  python scripts/run_finetune_qwen.py --train --eval
  python scripts/run_finetune_qwen.py --eval-only --max-eval 55
  python scripts/run_finetune_qwen.py --train --max-train 32   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm_finetune import FinetuneConfig, eval_finetuned, merge_with_baselines, train_lora  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true", help="eval on test after train")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--merge", action="store_true", help="write comparison with 4 baselines + finetuned")
    p.add_argument("--max-eval", type=int, default=55)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    if not (args.train or args.eval_only or args.merge):
        p.error("Pass --train, --eval-only, and/or --merge")

    cfg = FinetuneConfig(
        base_model_id=args.base_model,
        epochs=args.epochs,
        max_train_samples=args.max_train,
        use_4bit=not args.no_4bit,
    )

    if args.train:
        train_lora(ROOT, cfg)
    if args.eval or args.eval_only:
        eval_finetuned(ROOT, split="test", max_eval=args.max_eval, cfg=cfg)
    if args.merge:
        c = merge_with_baselines(ROOT)
        print("best", c.get("best_combo_id"), c.get("best_accuracy_pct"))


if __name__ == "__main__":
    main()
