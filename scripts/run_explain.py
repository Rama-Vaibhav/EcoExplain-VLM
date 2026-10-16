#!/usr/bin/env python3
"""Zero-shot justifications and leak-free one-word eval. Writes only under results/explain/.

  # Structured cite-only prompts (recommended for report):
  python scripts/run_explain.py --mode explain --prompt-style structured --combo qwen_vl_custom_mlp_qwen

  # Mac/laptop: default --laptop (2B Qwen, 0.5B LLaVA, 336px). ~15–20 min for 55 patches (Qwen).
  python scripts/run_explain.py --dry-run --limit 3 --prompt-style structured
  python scripts/score_explanations.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm_combos import (  # noqa: E402
    get_combo,
    load_t0_t1,
    records_for_split,
    resolve_checkpoint,
    text_has_label_leak,
)
from src.vlm_eval import (  # noqa: E402
    EXPLAIN_MAX_NEW_TOKENS,
    SHORT_EXPLAIN_COMBOS,
    explain_ask_suffix,
    generate_one,
    load_hf_model,
    parse_class,
    parse_explanation,
    unload,
)
from src.vlm_evidence import compact_evidence, compact_evidence_structured  # noqa: E402

DEFAULT_COMBOS = [
    "llava15_clip_mlp_vicuna",
    "paligemma_siglip_linear_gemma",
    "qwen_vl_custom_mlp_qwen",
    "blip2_clip_qformer_opt",
]


def _subset(root: Path, limit: int | None) -> list[dict]:
    records = records_for_split(root / "dataset.jsonl", root / "data" / "splits", "test")
    return records if limit is None else records[:limit]


def _assert_clean(rec: dict, text: str) -> None:
    leak = text_has_label_leak(text, rec.get("candidate_class"))
    if leak:
        raise AssertionError(f"{rec.get('id')} prompt still contains {leak}")


def dry_run(root: Path, combos: list[str], limit: int, prompt_style: str) -> None:
    records = _subset(root, limit)
    out_dir = root / "results" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = []
    for rec in records:
        table = (
            compact_evidence_structured(rec)
            if prompt_style == "structured"
            else compact_evidence(rec)
        )
        _assert_clean(rec, table)
        preview.append({"id": rec["id"], "gold": rec.get("candidate_class"), "evidence": table})
    path = out_dir / "dry_run_prompts.json"
    path.write_text(json.dumps({"n": len(preview), "combos": combos, "patches": preview}, indent=2))
    print(f"dry-run ok patches={len(preview)} combos={len(combos)} wrote {path}")
    for combo_id in combos:
        kind = "short" if combo_id in SHORT_EXPLAIN_COMBOS else "json"
        tokens = EXPLAIN_MAX_NEW_TOKENS.get(combo_id, 256)
        print(f"  {combo_id}: explain format={kind} max_new_tokens={tokens}")


def _run_combo(
    root: Path,
    combo_id: str,
    records: list[dict],
    mode: str,
    laptop: bool,
    prompt_style: str,
) -> None:
    combo = resolve_checkpoint(get_combo(combo_id), laptop=laptop)
    answer_format = "classify"
    max_new = 48
    prompt_mode = "evidence_only"
    if mode == "explain":
        answer_format = "short" if combo_id in SHORT_EXPLAIN_COMBOS else "json"
        max_new = EXPLAIN_MAX_NEW_TOKENS.get(combo_id, 256)
        prompt_mode = "explain"
    gen_kw = {"max_new_tokens": max_new, "do_sample": False}
    out_dir = root / "results" / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "explain" if mode == "explain" else "oneword"
    prefix = f"{name}_structured" if mode == "explain" and prompt_style == "structured" else name
    out_path = out_dir / f"{prefix}_{combo.combo_id}.json"

    processor, model, device = load_hf_model(combo)
    print(f"{combo.combo_id}: {mode} on {device} n={len(records)} format={answer_format}")
    rows = []
    t0 = time.time()
    try:
        for rec in records:
            images = list(load_t0_t1(rec, root))
            if mode == "explain":
                evidence = (
                    compact_evidence_structured(rec)
                    if prompt_style == "structured"
                    else compact_evidence(rec)
                )
            else:
                from src.vlm_combos import encoder_text

                evidence = encoder_text(rec, text_mode="evidence_only")
            _assert_clean(rec, evidence)
            sample = {
                "id": rec["id"],
                "images": images,
                "input_text": evidence,
                "label": rec.get("candidate_class"),
            }
            if mode == "explain":
                sample["ask_suffix"] = explain_ask_suffix(prompt_style, answer_format)
            started = time.time()
            try:
                raw = generate_one(
                    processor,
                    model,
                    sample,
                    gen_kw,
                    max_side=336 if laptop else None,
                    answer_format=answer_format,
                )
            except Exception as exc:
                raw = f"[generate_error] {exc}"
            parsed = parse_explanation(raw) if mode == "explain" else None
            pred = (parsed or {}).get("change_class") if parsed else parse_class(raw)
            if pred is None:
                pred = parse_class(raw)
            rows.append(
                {
                    "id": rec["id"],
                    "gold": rec.get("candidate_class"),
                    "pred": pred,
                    "correct": pred == rec.get("candidate_class"),
                    "raw_text": raw[:4000],
                    "parsed": parsed,
                    "prompt_mode": prompt_mode,
                    "answer_format": answer_format,
                    "seconds": round(time.time() - started, 2),
                }
            )
    finally:
        unload(model, processor)

    labeled = [r for r in rows if r["pred"] is not None]
    acc = sum(r["correct"] for r in labeled) / len(labeled) if labeled else None
    report = {
        "combo_id": combo.combo_id,
        "hf_model_id": combo.hf_model_id,
        "laptop": laptop,
        "mode": mode,
        "prompt_mode": prompt_mode,
        "prompt_style": prompt_style if mode == "explain" else None,
        "answer_format": answer_format,
        "generation": gen_kw,
        "n_eval": len(rows),
        "n_parsed_class": len(labeled),
        "accuracy": acc,
        "seconds": round(time.time() - t0, 2),
        "predictions": rows,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_path} accuracy={acc}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Evaluate the first N test patches. Use 3 to check outputs.")
    p.add_argument("--dry-run", action="store_true", help="Build prompts only. Does not load a model.")
    p.add_argument("--mode", choices=("explain", "oneword", "both"), default="both")
    p.add_argument("--combo", action="append", dest="combos", default=None)
    p.add_argument("--no-laptop", action="store_true", help="Use the full HF id (LLaVA 7B, Qwen 7B).")
    p.add_argument(
        "--prompt-style",
        choices=("structured", "legacy"),
        default="structured",
        help="Explain prompts: structured = cite-only from ALLOWED_FACTS (default); legacy = original ask.",
    )
    args = p.parse_args()
    combos = args.combos or list(DEFAULT_COMBOS)
    if args.dry_run:
        dry_run(ROOT, combos, args.limit or 3, args.prompt_style)
        return
    records = _subset(ROOT, args.limit)
    modes = ["explain", "oneword"] if args.mode == "both" else [args.mode]
    for combo_id in combos:
        for mode in modes:
            try:
                _run_combo(
                    ROOT,
                    combo_id,
                    records,
                    mode,
                    laptop=not args.no_laptop,
                    prompt_style=args.prompt_style,
                )
            except Exception as exc:
                print(f"{combo_id} {mode}: FAILED {exc}")


if __name__ == "__main__":
    main()
