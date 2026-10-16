#!/usr/bin/env python3
"""Score results/explain/explain_*.json. Does not read or write other results/ files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain_metrics import (  # noqa: E402
    distinct_2,
    justification_text,
    near_duplicate_share,
    rouge_l,
    score_justification,
)
from src.vlm_combos import load_jsonl  # noqa: E402
from src.vlm_evidence import compact_evidence  # noqa: E402


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def score_report(report: dict, records: dict[str, dict]) -> dict:
    per_patch = {}
    just_texts = []
    rouge_ev = []
    rouge_tmpl = []
    consistent = []
    claim_correct = 0
    claim_total = 0
    halluc = 0
    explained = 0
    for row in report.get("predictions") or []:
        rec = records.get(row["id"])
        if rec is None:
            continue
        parsed = row.get("parsed") or {}
        text = justification_text(parsed, row.get("raw_text") or "")
        stats = score_justification(text, rec, row.get("pred"))
        ev = rouge_l(text, compact_evidence(rec))
        tmpl = rouge_l(text, rec.get("explanation") or "")
        just_texts.append(text)
        rouge_ev.append(ev)
        rouge_tmpl.append(tmpl)
        if stats["self_consistent"] is not None:
            consistent.append(1.0 if stats["self_consistent"] else 0.0)
        claim_total += stats["n_claims"]
        claim_correct += stats["n_correct"]
        if text.strip():
            explained += 1
            if stats["has_wrong_claim"]:
                halluc += 1
        per_patch[row["id"]] = {
            "n_claims": stats["n_claims"],
            "n_correct": stats["n_correct"],
            "faithfulness": stats["faithfulness"],
            "self_consistent": stats["self_consistent"],
            "rouge_l_evidence": round(ev, 4),
            "rouge_l_template": round(tmpl, 4),
        }
    labeled = [r for r in report.get("predictions") or [] if r.get("pred") is not None]
    acc = sum(1 for r in labeled if r.get("correct")) / len(labeled) if labeled else None
    return {
        "combo_id": report.get("combo_id"),
        "hf_model_id": report.get("hf_model_id"),
        "prompt_mode": report.get("prompt_mode"),
        "prompt_style": report.get("prompt_style") or "legacy",
        "answer_format": report.get("answer_format"),
        "n_eval": report.get("n_eval"),
        "accuracy": acc,
        "mean_checkable_claims": (claim_total / explained) if explained else 0.0,
        "pct_correct_claims": (claim_correct / claim_total) if claim_total else None,
        "hallucination_rate": (halluc / explained) if explained else None,
        "self_consistency": _mean(consistent),
        "mean_rouge_l_evidence": _mean(rouge_ev),
        "mean_rouge_l_template": _mean(rouge_tmpl),
        "distinct_2": distinct_2(just_texts),
        "near_duplicate_share": near_duplicate_share(just_texts),
        "per_patch": per_patch,
    }


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def _select_reports(paths: list[Path], include_legacy: bool) -> list[dict]:
    """One report per combo_id: the structured run when it exists, else the legacy run.

    With include_legacy=True every explain report is kept (each row carries prompt_style).
    """
    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("mode") not in (None, "explain") or "predictions" not in report:
            continue
        report["prompt_style"] = report.get("prompt_style") or "legacy"
        report["_source_file"] = path.name
        reports.append(report)
    if include_legacy:
        return reports
    chosen: dict[str, dict] = {}
    for report in reports:
        cid = report.get("combo_id")
        current = chosen.get(cid)
        if current is None or (report["prompt_style"] == "structured" and current["prompt_style"] != "structured"):
            chosen[cid] = report
    return list(chosen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Score every explain_*.json. Default keeps one row per model, preferring explain_structured_*.",
    )
    args = parser.parse_args()
    explain_dir = ROOT / "results" / "explain"
    paths = sorted(explain_dir.glob("explain_*.json"))
    if not paths:
        raise SystemExit(f"No explain_*.json under {explain_dir}. Run scripts/run_explain.py first.")
    records = {r["id"]: r for r in load_jsonl(ROOT / "dataset.jsonl")}
    rows = []
    for report in _select_reports(paths, args.include_legacy):
        row = score_report(report, records)
        row["source_file"] = report["_source_file"]
        row["n_parsed_class"] = sum(1 for r in report["predictions"] if r.get("pred") is not None)
        rows.append(row)
    payload = {"n_models": len(rows), "models": rows}
    out_json = explain_dir / "explanation_scores.json"
    out_csv = explain_dir / "explanation_scores.csv"
    out_json.write_text(json.dumps(payload, indent=2))
    fields = [
        "combo_id",
        "prompt_style",
        "source_file",
        "n_eval",
        "n_parsed_class",
        "accuracy",
        "mean_checkable_claims",
        "pct_correct_claims",
        "hallucination_rate",
        "self_consistency",
        "mean_rouge_l_evidence",
        "mean_rouge_l_template",
        "distinct_2",
        "near_duplicate_share",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    header = (
        "| Model | Prompt | Accuracy | Checkable claims | Correct claims | Hallucination | "
        "Self-consistent | ROUGE-L vs evidence | ROUGE-L vs template | Distinct-2 | Near-duplicates |"
    )
    sep = "|" + "---|" * 11
    print(header)
    print(sep)
    for row in rows:
        print(
            f"| {row['combo_id']} | {row['prompt_style']} | {_pct(row['accuracy'])} | {row['mean_checkable_claims']:.2f} | "
            f"{_pct(row['pct_correct_claims'])} | {_pct(row['hallucination_rate'])} | "
            f"{_pct(row['self_consistency'])} | {_pct(row['mean_rouge_l_evidence'])} | "
            f"{_pct(row['mean_rouge_l_template'])} | {row['distinct_2']:.2f} | "
            f"{_pct(row['near_duplicate_share'])} |"
        )
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
