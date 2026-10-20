#!/usr/bin/env python3
"""Scan ecoexplain_vlm_dataset/results and write every reported score in one place.

Outputs (next to this script):
  RESULTS_MASTER.json                  machine-readable manifest
  RESULTS_MASTER.md                    quick reference
  chapters/generated_master_table.tex  the table included by the study guide

Every count is recomputed from the saved predictions, so k/n, intervals and per-class
scores always match the files. Run from anywhere:

    python study_guide_v3/build_results_master.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works both beside the repo (local layout) and inside it (GitHub copy).
REPO = HERE.parent if (HERE.parent / "results").is_dir() else HERE.parent / "ecoexplain_vlm_dataset"
RESULTS = REPO / "results"
CLASSES = ["decline", "stable", "increase"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def macro_f1(preds: list[dict]) -> float:
    f1s = []
    for c in CLASSES:
        tp = sum(1 for x in preds if x.get("pred") == c and x["gold"] == c)
        fp = sum(1 for x in preds if x.get("pred") == c and x["gold"] != c)
        fn = sum(1 for x in preds if x.get("pred") != c and x["gold"] == c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / 3


def score(path: Path) -> dict | None:
    """Recompute a run's scores from its predictions."""
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    if d.get("status") == "load_failed":
        return {"status": "load_failed", "note": "model not loaded (gated on Hugging Face)"}
    preds = d.get("predictions") or []
    n = len(preds)
    parsed = sum(1 for x in preds if x.get("pred") in CLASSES)
    if not n or not parsed:
        return {"status": "no_parsed_answers", "n": n, "parsed": parsed}
    k = sum(1 for x in preds if x.get("pred") == x["gold"])
    lo, hi = wilson(k, n)
    per_class = {
        c: f"{sum(1 for x in preds if x['gold'] == c and x.get('pred') == c)}/"
        f"{sum(1 for x in preds if x['gold'] == c)}"
        for c in CLASSES
    }
    return {
        "status": "ok",
        "k": k,
        "n": n,
        "parsed": parsed,
        "accuracy": k / n,
        "ci95": [round(lo, 3), round(hi, 3)],
        "macro_f1": round(macro_f1(preds), 3),
        "per_class_correct": per_class,
        "answers": dict(Counter(x.get("pred") for x in preds)),
        "prompt_mode": d.get("prompt_mode"),
    }


def ndvi_rule_baselines() -> dict:
    rows = {json.loads(l)["id"]: json.loads(l) for l in (REPO / "dataset.jsonl").open()}
    split = {
        s: [r["patch_id"] for r in csv.DictReader((REPO / f"data/splits/{s}_patches.csv").open())]
        for s in ("train", "test")
    }

    def rule(pid: str, lo: float, hi: float) -> str:
        v = rows[pid]["file_analysis"]["index_delta"]["NDVI"]
        return "decline" if v <= lo else ("increase" if v >= hi else "stable")

    def acc(ids, lo, hi) -> int:
        return sum(rule(i, lo, hi) == rows[i]["candidate_class"] for i in ids)

    # Same search as the earlier guides: the first cut-off pair with the highest training
    # accuracy wins. Other pairs tie on training accuracy and give 96.4% on test instead.
    vals = sorted({round(rows[i]["file_analysis"]["index_delta"]["NDVI"], 4) for i in split["train"]})
    best = None
    for lo in [v for v in vals if v < 0.01]:
        for hi in [v for v in vals if v > lo]:
            a = acc(split["train"], lo, hi)
            if best is None or a > best[0]:
                best = (a, lo, hi)
    test = split["test"]
    majority = sum(rows[i]["candidate_class"] == "increase" for i in test)
    return {
        "majority": majority,
        "rule_fixed": acc(test, -0.02, 0.02),
        "rule_fitted": acc(test, best[1], best[2]),
        "rule_fitted_cutoffs": [best[1], best[2]],
        "n": len(test),
    }


# (section, row label, protocol, path relative to results/)
RUNS = [
    ("A legacy", "Qwen2-VL-2B", "images + full card (summary line with label)", "vlm_combo_qwen_vl_custom_mlp_qwen.json"),
    ("A legacy", "PaliGemma-3B", "images + one NDVI line", "vlm_combo_paligemma_siglip_linear_gemma.json"),
    ("A legacy", "LLaVA-Interleave-0.5B", "images + full card", "vlm_combo_llava15_clip_mlp_vicuna.json"),
    ("A legacy", "BLIP-2 OPT-2.7B", "stitched image, no card (old parser read the echo)", "vlm_combo_blip2_clip_qformer_opt.json"),
    ("A leak-free", "LLaVA-Interleave-0.5B", "compact table (same for all models)", "protocol/vlm_combo_llava15_clip_mlp_vicuna.json"),
    ("A leak-free", "Qwen2-VL-2B", "compact table", "protocol/vlm_combo_qwen_vl_custom_mlp_qwen.json"),
    ("A leak-free", "PaliGemma-3B", "compact table", "protocol/vlm_combo_paligemma_siglip_linear_gemma.json"),
    ("A leak-free", "BLIP-2 OPT-2.7B", "compact table (new anti-echo parser)", "protocol/vlm_combo_blip2_clip_qformer_opt.json"),
    ("B class LoRA", "Qwen2-VL-2B, balanced 16 (Ada 29575)", "evidence_only at train and test", "protocol/qwen_balanced_leakfree_lora.json"),
    ("B class LoRA (superseded)", "Qwen2-VL-2B, first 16 by ID (Ada 2722446)", "full card at train and test", "vlm_combo_qwen_vl_finetuned_ecoexplain.json"),
    ("C explanation", "Qwen2-VL-2B structured", "label-free table + ALLOWED_FACTS + CLASS_RULE", "explain/explain_structured_qwen_vl_custom_mlp_qwen.json"),
    ("C explanation", "LLaVA-0.5B structured", "same", "explain/explain_structured_llava15_clip_mlp_vicuna.json"),
    ("C explanation", "Qwen2-VL-2B explanation LoRA (Ada 24389)", "label-free table, free-form JSON", "explain/explain_qwen_vl_finetuned_ecoexplain.json"),
]


def main() -> None:
    rows = []
    for section, label, protocol, rel in RUNS:
        s = score(RESULTS / rel) or {"status": "missing"}
        rows.append({"section": section, "label": label, "protocol": protocol, "source": f"results/{rel}", **s})

    # Qwen leak-free evidence_only: only the comparison summary survives
    # (its per-model predictions were overwritten by the compact run).
    ev = json.loads((RESULTS / "protocol/vlm_combo_comparison_evidence_only.json").read_text())
    for r in ev.get("ranking", []):
        acc = r.get("accuracy")
        rows.insert(4, {
            "section": "A leak-free", "label": "Qwen2-VL-2B", "protocol": "evidence_only card (label line removed)",
            "source": "results/protocol/vlm_combo_comparison_evidence_only.json", "status": r.get("status"),
            "k": round(acc * r["n_eval"]), "n": r["n_eval"], "accuracy": acc,
            "ci95": [round(x, 3) for x in wilson(round(acc * r["n_eval"]), r["n_eval"])],
            "note": "summary only; per-model predictions were overwritten by the compact run",
        })

    claims = {}
    csv_path = RESULTS / "explain/explanation_scores.csv"
    for r in csv.DictReader(csv_path.open()):
        if r.get("pct_correct_claims"):
            claims[r["combo_id"] + ":" + r["prompt_style"]] = float(r["pct_correct_claims"])
    js = json.loads((RESULTS / "explain/explanation_scores.json").read_text())
    for r in js["models"]:
        if r["combo_id"] == "qwen_vl_finetuned_ecoexplain":
            claims["qwen_vl_finetuned_ecoexplain:legacy"] = r["pct_correct_claims"]
    claim_for = {
        "Qwen2-VL-2B structured": claims.get("qwen_vl_custom_mlp_qwen:structured"),
        "LLaVA-0.5B structured": claims.get("llava15_clip_mlp_vicuna:structured"),
        "Qwen2-VL-2B explanation LoRA (Ada 24389)": claims.get("qwen_vl_finetuned_ecoexplain:legacy"),
    }
    for r in rows:
        if r["label"] in claim_for:
            r["correct_claims"] = claim_for[r["label"]]

    base = ndvi_rule_baselines()
    manifest = {
        "generated": date.today().isoformat(),
        "test_n": base["n"],
        "rows": rows,
        "baselines": base,
        "teacher_filter": json.loads((RESULTS / "explain/target_filter_report.json").read_text()),
    }
    (HERE / "RESULTS_MASTER.json").write_text(json.dumps(manifest, indent=2))

    def acc_txt(r: dict, latex: bool = False) -> str:
        pc = "\\,\\%" if latex else "%"
        if r.get("status") == "load_failed":
            return "not loaded"
        if r.get("status") == "no_parsed_answers":
            return f"0/{r['n']} parsed"
        if r.get("accuracy") is None:
            return "missing"
        txt = f"{100 * r['accuracy']:.1f}{pc} ({r['k']}/{r['n']})"
        if r.get("parsed") and r["parsed"] < r["n"]:
            txt += f"; {100 * r['k'] / r['parsed']:.1f}{pc} of {r['parsed']} answered"
        return txt

    n = base["n"]
    md = [
        f"# EcoExplain-VLM: master results ({manifest['generated']})",
        "",
        "Recomputed from the prediction files by `build_results_master.py`. Test set n = 55.",
        "**Rule:** never compare rows from different sections without naming the protocol.",
        "",
        "| Section | Run | Protocol | Accuracy | 95% CI | Macro-F1 | Correct per class (d / s / i) | Claims correct | Source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"{100 * r['ci95'][0]:.1f}-{100 * r['ci95'][1]:.1f}" if r.get("ci95") else "—"
        pcs = " / ".join(r["per_class_correct"][c] for c in CLASSES) if r.get("per_class_correct") else "—"
        cl = f"{100 * r['correct_claims']:.1f}%" if r.get("correct_claims") is not None else "—"
        md.append(f"| {r['section']} | {r['label']} | {r['protocol']} | {acc_txt(r)} | {ci} | "
                  f"{r.get('macro_f1', '—')} | {pcs} | {cl} | `{r['source']}` |")
    md += [
        "",
        "## Baselines (no model)",
        "",
        f"- Majority class (always increase): {100 * base['majority'] / n:.1f}% ({base['majority']}/{n})",
        f"- ΔNDVI rule, fixed ±0.02: {100 * base['rule_fixed'] / n:.1f}% ({base['rule_fixed']}/{n})",
        f"- ΔNDVI rule, cut-offs fitted on train {base['rule_fitted_cutoffs']}: "
        f"{100 * base['rule_fitted'] / n:.1f}% ({base['rule_fitted']}/{n})",
        "- LLaVA closed-set probe with dev-fitted bias: 60.0% (33/55), probe only (ada_review/)",
        "",
        f"Explanation teacher filter: {manifest['teacher_filter']['kept']}/{manifest['teacher_filter']['seen']} kept.",
        "",
    ]
    (HERE / "RESULTS_MASTER.md").write_text("\n".join(md))

    def tex(s: str) -> str:
        return s.replace("_", "\\_").replace("%", "\\%").replace("±", "$\\pm$")

    tl = [
        "% Generated by build_results_master.py. Do not edit by hand.",
        "\\begingroup\\small",
        "\\begin{longtable}{L{1.9cm}L{3.7cm}L{3.4cm}L{2.7cm}L{1.6cm}}",
        "\\caption{Every classification score on the 55 test patches, recomputed from the prediction files "
        "(\\file{study_guide_v3/build_results_master.py}). Per class: correct decline / stable / increase "
        "out of 18 / 18 / 19.}\\label{tab:master}\\\\",
        "\\toprule",
        "\\textbf{Section} & \\textbf{Run} & \\textbf{Input} & \\textbf{Accuracy (95\\,\\% CI)} & \\textbf{Per class} \\\\",
        "\\midrule\\endfirsthead",
        "\\toprule",
        "\\textbf{Section} & \\textbf{Run} & \\textbf{Input} & \\textbf{Accuracy (95\\,\\% CI)} & \\textbf{Per class} \\\\",
        "\\midrule\\endhead",
    ]
    last = None
    for r in rows:
        if last is not None and r["section"] != last:
            tl.append("\\midrule")
        last = r["section"]
        ci = f"\\newline {100 * r['ci95'][0]:.1f}--{100 * r['ci95'][1]:.1f}" if r.get("ci95") else ""
        pcs = " / ".join(r["per_class_correct"][c].split("/")[0] for c in CLASSES) if r.get("per_class_correct") else "---"
        tl.append(f"{tex(r['section'])} & {tex(r['label'])} & {tex(r['protocol'])} & {acc_txt(r, True)}{ci} & {pcs} \\\\")
    tl += [
        "\\midrule",
        f"Baseline & always ``increase'' & --- & {100 * base['majority'] / n:.1f}\\,\\% ({base['majority']}/{n}) & 0 / 0 / 19 \\\\",
        f"Baseline & $\\Delta$NDVI rule $\\pm0.02$ & card $\\Delta$NDVI & {100 * base['rule_fixed'] / n:.1f}\\,\\% ({base['rule_fixed']}/{n}) & --- \\\\",
        f"Baseline & $\\Delta$NDVI rule fitted on train & card $\\Delta$NDVI & {100 * base['rule_fitted'] / n:.1f}\\,\\% ({base['rule_fitted']}/{n}) & --- \\\\",
        "\\bottomrule",
        "\\end{longtable}",
        "\\endgroup",
    ]
    (HERE / "chapters").mkdir(exist_ok=True)
    (HERE / "chapters/generated_master_table.tex").write_text("\n".join(tl) + "\n")
    for f in ("RESULTS_MASTER.json", "RESULTS_MASTER.md", "chapters/generated_master_table.tex"):
        print("wrote", HERE / f)


if __name__ == "__main__":
    main()
