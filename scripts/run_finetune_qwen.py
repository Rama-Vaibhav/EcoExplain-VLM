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
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--max-train", type=int, default=128, help="cap train patches (anti-overfit)")
    p.add_argument(
        "--eval-prompt",
        choices=("minimal", "evidence_only"),
        default="evidence_only",
        help="Test prompt. 'full' is rejected because it contains candidate_class.",
    )
    p.add_argument(
        "--target",
        choices=("class", "explain"),
        default="class",
        help="explain trains on data/explain_targets/train_targets.jsonl and writes results/explain/ only.",
    )
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--eval-dev", action="store_true", help="also eval on dev after train")
    args = p.parse_args()

    if not (args.train or args.eval_only or args.merge):
        p.error("Pass --train, --eval-only, and/or --merge")

    cfg = FinetuneConfig(
        base_model_id=args.base_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        max_train_samples=args.max_train,
        use_4bit=not args.no_4bit,
        early_stopping=not args.no_early_stop,
        early_stopping_patience=args.patience,
        target_mode=args.target,
        train_text_mode="evidence_only",
    )
    if args.target == "explain":
        cfg.adapter_dir = "results/checkpoints/qwen2vl_lora_explain"

    if args.train:
        train_lora(ROOT, cfg)
    if args.target == "explain" and (args.eval or args.eval_only):
        from src.vlm_finetune import eval_explain_finetuned

        eval_explain_finetuned(ROOT, max_eval=args.max_eval, cfg=cfg)
        return
    if args.eval_dev and not args.eval_only:
        # Separate file so the later test eval does not overwrite dev accuracy.
        eval_finetuned(
            ROOT,
            split="dev",
            max_eval=55,
            cfg=cfg,
            text_mode=args.eval_prompt,
            combo_id_suffix="_dev",
        )
    if args.eval or args.eval_only:
        eval_finetuned(
            ROOT,
            split="test",
            max_eval=args.max_eval,
            cfg=cfg,
            text_mode=args.eval_prompt,
        )
    if args.merge:
        c = merge_with_baselines(ROOT)
        print("best", c.get("best_combo_id"), c.get("best_accuracy_pct"))


if __name__ == "__main__":
    main()
