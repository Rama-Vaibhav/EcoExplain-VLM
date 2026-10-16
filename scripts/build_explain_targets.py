#!/usr/bin/env python3
"""Write train-split explanation targets with Qwen2-VL-7B. Run this on Ada, not the laptop.

Keeps a patch only when the 7B class matches gold, faithfulness is 1.0, and there are
at least two checkable claims. The rule-based dataset explanation is not used as the target.

  python scripts/build_explain_targets.py --split train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain_metrics import justification_text, score_justification  # noqa: E402
from src.vlm_combos import load_t0_t1, records_for_split, text_has_label_leak  # noqa: E402
from src.vlm_eval import generate_one, parse_explanation, unload  # noqa: E402
from src.vlm_evidence import compact_evidence  # noqa: E402

# 7B does not fit a typical Ada home quota. Override with --teacher Qwen/Qwen2-VL-7B-Instruct
# only if the weights already live on a scratch disk.
DEFAULT_TEACHER = "Qwen/Qwen2-VL-2B-Instruct"


def load_teacher(model_id: str):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("teacher", model_id)
    return processor, model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    args = p.parse_args()
    records = records_for_split(ROOT / "dataset.jsonl", ROOT / "data" / "splits", args.split)
    if args.limit is not None:
        records = records[: args.limit]
    processor, model = load_teacher(args.teacher)
    gen_kw = {"max_new_tokens": 256, "do_sample": False}
    kept = []
    seen = 0
    try:
        for rec in records:
            seen += 1
            evidence = compact_evidence(rec)
            leak = text_has_label_leak(evidence, rec.get("candidate_class"))
            if leak:
                raise AssertionError(f"{rec.get('id')} prompt contains {leak}")
            sample = {
                "id": rec["id"],
                "images": list(load_t0_t1(rec, ROOT)),
                "input_text": evidence,
                "label": rec.get("candidate_class"),
            }
            try:
                raw = generate_one(processor, model, sample, gen_kw, answer_format="json")
            except Exception as exc:
                print("skip", rec["id"], exc)
                continue
            parsed = parse_explanation(raw)
            pred = parsed.get("change_class")
            text = justification_text(parsed, raw)
            stats = score_justification(text, rec, pred)
            ok = (
                pred == rec.get("candidate_class")
                and stats["n_claims"] >= 2
                and stats["faithfulness"] == 1.0
            )
            if not ok:
                continue
            target = {
                "change_class": pred,
                "visual_change": parsed.get("visual_change") or "",
                "justification": parsed.get("justification") or "",
                "supporting_evidence": parsed.get("supporting_evidence") or [],
                "conflicting_evidence": parsed.get("conflicting_evidence") or [],
                "possible_driver": parsed.get("possible_driver") or "",
            }
            kept.append(
                {
                    "id": rec["id"],
                    "change_class": pred,
                    "justification": target["justification"],
                    "n_claims": stats["n_claims"],
                    "target": json.dumps(target, ensure_ascii=False),
                }
            )
            if seen % 10 == 0:
                print(f"seen={seen} kept={len(kept)}", flush=True)
    finally:
        unload(model, processor)

    out = ROOT / "data" / "explain_targets" / f"{args.split}_targets.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    summary = {
        "split": args.split,
        "teacher": args.teacher,
        "seen": seen,
        "kept": len(kept),
        "path": str(out),
    }
    report = ROOT / "results" / "explain" / "target_filter_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    print(f"survived {len(kept)} / {seen}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
