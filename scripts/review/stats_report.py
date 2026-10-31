#!/usr/bin/env python3
"""C0.3/C0.4/C0.6 statistics for every reported run (sir's Major 2, 3.5, 5.5, 6; statistical checks).

For each run: accuracy with Wilson 95% CI over all test patches, answered n, macro-F1,
per-class precision/recall/F1 and the confusion matrix. Baselines on both splits (constant
predictors, fixed and fitted ΔNDVI rules, the label rule). Paired exact two-sided McNemar tests
with Holm adjustment over every test reported in the paper. Sensitivity re-scoring on the test
patches whose label follows the stated rule and without the edge-clipped patches.

The McNemar family below was fixed on 31 Oct 2026 before the matched-control runs finished.
Runs whose file does not exist yet are reported as "pending" and left out of the family.

    .venv/bin/python scripts/review/stats_report.py
Outputs: results/review/stats_report.json, results/review/stats_report.md,
         report/paper_v3/v6_tables/*.tex (generated; do not edit by hand)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASSES, RESULTS, REVIEW, ROOT, card_dndvi, holm, mcnemar_exact, rows, rule, split_ids,
    summarise, write_json,
)

TEX_DIR = ROOT.parent / "report" / "paper_v3" / "v6_tables"

# (key, track, label, protocol, split, file under results/)
RUNS = [
    ("qwen_full", "A", "Qwen2-VL-2B", "full card incl. summary line, 336 px", "random", "vlm_combo_qwen_vl_custom_mlp_qwen.json"),
    ("qwen_full_rerun", "A", "Qwen2-VL-2B", "full card incl. summary line, 336 px (31 Oct re-run)", "random", "protocol/random_full_card/vlm_combo_qwen_vl_custom_mlp_qwen_336_rerun.json"),
    ("qwen_compact_repro", "A", "Qwen2-VL-2B", "compact table, 336 px (31 Oct re-run)", "random", "protocol/repro_gate/vlm_combo_qwen_vl_custom_mlp_qwen_qwen_repro336.json"),
    ("llava_full", "A", "LLaVA-Interleave-0.5B", "full card incl. summary line, 336 px", "random", "vlm_combo_llava15_clip_mlp_vicuna.json"),
    ("pali_ndvi", "A", "PaliGemma-3B", "card NDVI line only (adapted), 224 px", "random", "vlm_combo_paligemma_siglip_linear_gemma.json"),
    ("blip_img", "A", "BLIP-2 OPT-2.7B", "stitched images, no card (adapted); old parser read the echo", "random", "vlm_combo_blip2_clip_qformer_opt.json"),
    ("qwen_evid_336", "A", "Qwen2-VL-2B", "evidence-only card, 336 px", "random", "protocol/random_evidence_only/vlm_combo_qwen_vl_custom_mlp_qwen_336.json"),
    ("qwen_evid_384", "A", "Qwen2-VL-2B", "evidence-only card, 384 px", "random", "protocol/random_evidence_only/vlm_combo_qwen_vl_custom_mlp_qwen_384.json"),
    ("qwen_compact", "A", "Qwen2-VL-2B", "compact table, 336 px", "random", "protocol/random_compact/vlm_combo_qwen_vl_custom_mlp_qwen.json"),
    ("llava_compact", "A", "LLaVA-Interleave-0.5B", "compact table, 336 px", "random", "protocol/random_compact/vlm_combo_llava15_clip_mlp_vicuna.json"),
    ("llava_compact_repro", "A", "LLaVA-Interleave-0.5B", "compact table, 336 px (31 Oct re-run)", "random", "protocol/repro_gate/vlm_combo_llava15_clip_mlp_vicuna_repro336.json"),
    ("blip_compact", "A", "BLIP-2 OPT-2.7B", "compact table, stitched images", "random", "protocol/random_compact/vlm_combo_blip2_clip_qformer_opt.json"),
    ("class_lora", "B", "Qwen2-VL-2B class LoRA (job 29575)", "evidence-only, 384 px, 16 balanced", "random", "protocol/qwen_balanced_leakfree_lora.json"),
    ("class_lora_rep", "B", "Qwen2-VL-2B class LoRA (job 44523, repeat)", "evidence-only, 384 px, same 16", "random", "protocol/class_lora_repeats/qwen_balanced_leakfree_lora_job44523.json"),
    ("class_lora_fullcard", "B", "Qwen2-VL-2B class LoRA, first 16 by ID (superseded)", "full card at train and test", "random", "vlm_combo_qwen_vl_finetuned_ecoexplain.json"),
    ("llava_freeform", "C", "LLaVA-Interleave-0.5B", "free-form JSON explanation, 336 px", "random", "explain/explain_llava15_clip_mlp_vicuna.json"),
    ("qwen_struct", "C", "Qwen2-VL-2B", "structured cite-only, 256 tokens, 336 px", "random", "explain/explain_structured_qwen_vl_custom_mlp_qwen.json"),
    ("qwen_struct_512", "C", "Qwen2-VL-2B", "structured cite-only, 512 tokens, 336 px", "random", "explain/explain_structured_qwen_vl_custom_mlp_qwen_tok512.json"),
    ("llava_struct", "C", "LLaVA-Interleave-0.5B", "structured cite-only, 256 tokens, 336 px", "random", "explain/explain_structured_llava15_clip_mlp_vicuna.json"),
    ("qwen_expl_zs_336", "C", "Qwen2-VL-2B zero-shot", "free-form JSON explanation, 336 px", "random", "explain/explain_qwen_vl_custom_mlp_qwen.json"),
    ("qwen_expl_zs_384", "C", "Qwen2-VL-2B zero-shot", "free-form JSON explanation, 384 px", "random", "explain/explain_qwen_vl_custom_mlp_qwen_384.json"),
    ("expl_lora", "C", "Qwen2-VL-2B explanation LoRA (job 24389)", "free-form JSON explanation, 384 px", "random", "explain/explain_qwen_vl_finetuned_ecoexplain.json"),
    ("sp_qwen_compact", "S", "Qwen2-VL-2B", "compact table, 336 px", "spatial", "protocol/spatial_compact/vlm_combo_qwen_vl_custom_mlp_qwen.json"),
    ("sp_llava_compact", "S", "LLaVA-Interleave-0.5B", "compact table, 336 px", "spatial", "protocol/spatial_compact/vlm_combo_llava15_clip_mlp_vicuna.json"),
    ("sp_blip_compact", "S", "BLIP-2 OPT-2.7B", "compact table", "spatial", "protocol/spatial_compact/vlm_combo_blip2_clip_qformer_opt.json"),
]

# Every paired test reported in V6 (one Holm family). "const" = best constant predictor on that test set.
FAMILY = [
    # Both arms run on 31 Oct in the same environment (decision D11, fixed before either result existed).
    ("leak_primary", "qwen_full_rerun", "qwen_evid_336", "RQ1 primary leakage test: only the summary line and closing class-list instruction differ"),
    ("leak_secondary", "qwen_full", "qwen_compact", "RQ1 secondary: summary line and layout differ"),
    ("llava_vs_const", "llava_compact", "const_random", "RQ1 equal-input: best leak-free model vs constant"),
    ("classlora_vs_matched", "class_lora", "qwen_evid_384", "RQ2 class LoRA vs matched zero-shot (evidence-only, 384 px)"),
    ("classlora_vs_const", "class_lora", "const_random", "RQ2 class LoRA vs constant"),
    ("expllora_vs_matched", "expl_lora", "qwen_expl_zs_384", "RQ3 explanation LoRA vs matched zero-shot (same prompt, 384 px)"),
    ("expllora_vs_const", "expl_lora", "const_random", "RQ3 explanation LoRA vs constant"),
    ("sp_llava_vs_const", "sp_llava_compact", "const_spatial", "Spatial: LLaVA compact vs constant"),
    ("sp_qwen_vs_const", "sp_qwen_compact", "const_spatial", "Spatial: Qwen compact vs constant"),
]


def baselines(recs: dict, scheme: str, test: list[str], gold: list[str]) -> tuple[dict, dict]:
    train = split_ids("train", scheme)
    pool = pd.read_csv(ROOT / "data" / "raw_metadata" / "composite_stats.csv").set_index("patch_id")
    q15, q85 = pool.ndvi_change.quantile(0.15), pool.ndvi_change.quantile(0.85)
    dil = pd.read_csv(REVIEW / "card_dilution.csv").set_index("patch_id")

    def fit(values: dict[str, float]) -> tuple[int, list[tuple[float, float]]]:
        vals = sorted({round(values[i], 4) for i in train})
        best, ties = -1, []
        for lo in [v for v in vals if v < 0.01]:
            for hi in [v for v in vals if v > lo]:
                a = sum(rule(values[i], lo, hi) == recs[i]["candidate_class"] for i in train)
                if a > best:
                    best, ties = a, [(lo, hi)]
                elif a == best:
                    ties.append((lo, hi))
        return best, ties

    card = {i: card_dndvi(recs[i]) for i in recs}
    true = {i: float(dil.true_dNDVI[i]) for i in recs}
    preds: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    const = max(CLASSES, key=gold.count)
    train_major = max(CLASSES, key=[recs[i]["candidate_class"] for i in train].count)
    preds["const"] = [const] * len(test)
    meta["const"] = {"what": f"best constant predictor on this test set (always {const})"}
    preds["train_majority"] = [train_major] * len(test)
    meta["train_majority"] = {"what": f"training-set majority (always {train_major})"}
    preds["fixed_card"] = [rule(card[i], -0.02, 0.02) for i in test]
    meta["fixed_card"] = {"what": "fixed +-0.02 on card dNDVI (the structured prompt's CLASS_RULE)"}
    preds["label_rule_ee"] = [rule(float(pool.ndvi_change[i]), q15, q85) for i in test]
    meta["label_rule_ee"] = {"what": "Eq. label (batch-1 q15/q85) on the Earth Engine dNDVI that made the labels"}
    preds["label_rule_card"] = [rule(card[i], q15, q85) for i in test]
    meta["label_rule_card"] = {"what": "Eq. label (batch-1 q15/q85) on card dNDVI"}
    for name, values in (("fitted_card", card), ("fitted_true", true)):
        best, ties = fit(values)
        lo, hi = ties[0]
        preds[name] = [rule(values[i], lo, hi) for i in test]
        tie_acc = sorted({sum(rule(values[i], a, b) == g for i, g in zip(test, gold)) for a, b in ties})
        meta[name] = {"what": f"two cut-offs fitted on {scheme} train ({'card' if name == 'fitted_card' else 'true patch-mean'} dNDVI), first best pair kept",
                      "cutoffs": [lo, hi], "train_correct": best, "train_n": len(train),
                      "n_tied_pairs": len(ties), "tied_test_correct": tie_acc,
                      "misses": [{"id": i, "gold": g, "pred": p, "card": round(card[i], 4), "ee": round(float(pool.ndvi_change[i]), 4)}
                                 for i, g, p in zip(test, gold, preds[name]) if g != p]}
    return preds, meta


def main() -> None:
    recs = rows()
    out = {"runs": {}, "baselines": {}, "mcnemar": [], "sensitivity": {}}
    preds_by_key: dict[str, list] = {}
    for scheme in ("random", "spatial"):
        test = split_ids("test", scheme)
        gold = [recs[i]["candidate_class"] for i in test]
        bp, bmeta = baselines(recs, scheme, test, gold)
        for name, p in bp.items():
            out["baselines"][f"{scheme}:{name}"] = {**bmeta[name], **summarise(gold, p)}
        preds_by_key[f"const_{scheme}"] = bp["const"]
        for key, track, label, protocol, split, rel in RUNS:
            if split != scheme:
                continue
            if not (RESULTS / rel).exists():
                out["runs"][key] = {"status": "pending", "file": f"results/{rel}"}
                continue
            data = json.loads((RESULTS / rel).read_text())
            if data.get("status") == "load_failed":
                out["runs"][key] = {"status": "not loaded", "file": f"results/{rel}"}
                continue
            p = [{x["id"]: x.get("pred") for x in data["predictions"]}[i] for i in test]
            preds_by_key[key] = p
            out["runs"][key] = {"status": "ok", "track": track, "label": label, "protocol": protocol, "split": split,
                                "file": f"results/{rel}", "max_side": data.get("max_side"),
                                "hf_model_id": data.get("hf_model_id"), "generation": data.get("generation"),
                                **summarise(gold, p)}
        gold_by = {scheme: gold}
        out.setdefault("_gold", {}).update({scheme: dict(zip(test, gold))})

    # McNemar family.
    raw = {}
    for name, a, b, what in FAMILY:
        if a not in preds_by_key or b not in preds_by_key:
            out["mcnemar"].append({"test": name, "a": a, "b": b, "what": what, "status": "pending"})
            continue
        scheme = "spatial" if a.startswith("sp_") else "random"
        gold = list(out["_gold"][scheme].values())
        pa, pb = preds_by_key[a], preds_by_key[b]
        a_only = sum(x == g != y for x, y, g in zip(pa, pb, gold))
        b_only = sum(y == g != x for x, y, g in zip(pa, pb, gold))
        p = mcnemar_exact(a_only, b_only)
        raw[name] = p
        out["mcnemar"].append({"test": name, "a": a, "b": b, "what": what, "a_right_b_wrong": a_only,
                               "b_right_a_wrong": b_only, "p_exact_two_sided": p})
    adj = holm(raw)
    for row in out["mcnemar"]:
        if row["test"] in adj:
            row["p_holm"] = adj[row["test"]]
    out["holm_family_size"] = len(raw)

    # Sensitivity: rule-consistent test labels; without the three edge-clipped test patches.
    test = split_ids("test")
    gold = [recs[i]["candidate_class"] for i in test]
    off = set(pd.read_csv(REVIEW / "off_rule_labels.csv").patch_id)
    clipped = {"p00052", "p00086", "p04399"}
    for subset_name, keep in (("rule_consistent", [i not in off for i in test]),
                              ("unclipped", [i not in clipped for i in test])):
        g = [x for x, k in zip(gold, keep) if k]
        res = {}
        for key, p in preds_by_key.items():
            if key.startswith("sp_") or key == "const_spatial":
                continue
            pp = [x for x, k in zip(p, keep) if k]
            res[key] = {"k": sum(a == b for a, b in zip(g, pp)), "n": len(g)}
        out["sensitivity"][subset_name] = res
        pairs = {}
        for name, a, b in (("leak_primary", "qwen_full_rerun", "qwen_evid_336"),
                           ("llava_vs_const", "llava_compact", "const_random"),
                           ("classlora_vs_matched", "class_lora", "qwen_evid_384"),
                           ("expllora_vs_matched", "expl_lora", "qwen_expl_zs_384"),
                           ("expllora_vs_const", "expl_lora", "const_random")):
            if a not in preds_by_key or b not in preds_by_key:
                continue
            pa = [x for x, k in zip(preds_by_key[a], keep) if k]
            pb = [x for x, k in zip(preds_by_key[b], keep) if k]
            x = sum(u == gg != v for u, v, gg in zip(pa, pb, g))
            y = sum(v == gg != u for u, v, gg in zip(pa, pb, g))
            pairs[name] = {"a_right_b_wrong": x, "b_right_a_wrong": y, "p_exact_unadjusted": mcnemar_exact(x, y)}
        out["sensitivity"][subset_name + "_pairs"] = pairs
    out.pop("_gold")

    write_json("stats_report.json", out)
    write_markdown(out)
    write_tex(out)
    print("wrote results/review/stats_report.{json,md} and", TEX_DIR)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def write_markdown(out: dict) -> None:
    lines = ["# Statistics report (generated by scripts/review/stats_report.py)", "",
             "| Run | Split | Protocol | Acc (k/n) | Wilson 95% | Answered | Macro-F1 | Recall d / s / i |",
             "|---|---|---|---|---|---|---|---|"]
    for key, r in out["runs"].items():
        if r["status"] != "ok":
            lines.append(f"| {key} | | | {r['status']} | | | | |")
            continue
        rc = " / ".join(f"{r['confusion'][c][c]}/{r['per_class'][c]['support']}" for c in CLASSES)
        lines.append(f"| {r['label']} | {r['split']} | {r['protocol']} | {pct(r['accuracy'])} ({r['k']}/{r['n']}) | "
                     f"{pct(r['ci95'][0])}-{pct(r['ci95'][1])} | {r['answered']}/{r['n']} | {r['macro_f1']:.3f} | {rc} |")
    lines += ["", "## Baselines", "", "| Split:baseline | What | Acc (k/n) | Wilson 95% | Macro-F1 | Extra |", "|---|---|---|---|---|---|"]
    for key, b in out["baselines"].items():
        extra = {k: b[k] for k in ("cutoffs", "n_tied_pairs", "tied_test_correct") if k in b}
        lines.append(f"| {key} | {b['what']} | {pct(b['accuracy'])} ({b['k']}/{b['n']}) | {pct(b['ci95'][0])}-{pct(b['ci95'][1])} | {b['macro_f1']:.3f} | {extra} |")
    lines += ["", f"## McNemar (exact, two-sided; Holm over {out['holm_family_size']} tests)", "",
              "| Test | A | B | A right, B wrong | B right, A wrong | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for m in out["mcnemar"]:
        if m.get("status") == "pending":
            lines.append(f"| {m['test']} | {m['a']} | {m['b']} | pending | | | |")
        else:
            lines.append(f"| {m['test']} | {m['a']} | {m['b']} | {m['a_right_b_wrong']} | {m['b_right_a_wrong']} | "
                         f"{m['p_exact_two_sided']:.4f} | {m['p_holm']:.4f} |")
    lines += ["", "## Sensitivity (random test)", "", "| Run | 49 rule-consistent | 52 unclipped |", "|---|---|---|"]
    rc, uc = out["sensitivity"]["rule_consistent"], out["sensitivity"]["unclipped"]
    for key in rc:
        lines.append(f"| {key} | {rc[key]['k']}/{rc[key]['n']} | {uc[key]['k']}/{uc[key]['n']} |")
    (REVIEW / "stats_report.md").write_text("\n".join(lines) + "\n")


def write_tex(out: dict) -> None:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    head = "% Generated by ecoexplain_vlm_dataset/scripts/review/stats_report.py -- do not edit by hand.\n"
    order = ["qwen_full", "qwen_full_rerun", "qwen_evid_336", "qwen_compact", "llava_full", "llava_compact", "pali_ndvi",
             "class_lora", "class_lora_rep", "qwen_evid_384", "qwen_struct", "llava_struct", "llava_freeform",
             "qwen_expl_zs_384", "expl_lora", "sp_qwen_compact", "sp_llava_compact"]
    names = {"qwen_full": "Qwen, full card", "qwen_full_rerun": "Qwen, full card (31 Oct re-run)",
             "qwen_evid_336": "Qwen, evidence-only (336)", "qwen_compact": "Qwen, compact",
             "llava_full": "LLaVA, full card", "llava_compact": "LLaVA, compact", "pali_ndvi": "PaliGemma, NDVI line",
             "class_lora": "Class LoRA (29575)", "class_lora_rep": "Class LoRA (44523)", "qwen_evid_384": "Qwen, evidence-only (384)",
             "qwen_struct": "Qwen, structured", "llava_struct": "LLaVA, structured", "llava_freeform": "LLaVA, free-form",
             "qwen_expl_zs_384": "Qwen zero-shot, free-form (384)", "expl_lora": "Explanation LoRA",
             "sp_qwen_compact": "Spatial: Qwen, compact", "sp_llava_compact": "Spatial: LLaVA, compact"}
    body = []
    for key in order:
        r = out["runs"].get(key, {})
        if r.get("status") != "ok":
            continue
        pc = r["per_class"]
        cells = " & ".join(f"{pc[c]['precision']:.2f}/{pc[c]['recall']:.2f}/{pc[c]['f1']:.2f}" for c in CLASSES)
        body.append(f"{names[key]} & {r['k']}/{r['n']} & {r['answered']} & {r['macro_f1']:.2f} & {cells} \\\\")
    (TEX_DIR / "per_class_rows.tex").write_text(head + "\n".join(body) + "\n")

    conf = []
    for key in order:
        r = out["runs"].get(key, {})
        if r.get("status") != "ok":
            continue
        m = r["confusion"]
        cells = " & ".join(" ".join(str(m[g][p]) for p in (*CLASSES, "none")) for g in CLASSES)
        conf.append(f"{names[key]} & {cells} \\\\")
    (TEX_DIR / "confusion_rows.tex").write_text(head + "\n".join(conf) + "\n")

    mc = []
    labels = {"leak_primary": "Qwen full card vs evidence-only (both 31 Oct)", "leak_secondary": "Qwen full card vs compact",
              "llava_vs_const": "LLaVA compact vs constant", "classlora_vs_matched": "Class LoRA vs zero-shot (384)",
              "classlora_vs_const": "Class LoRA vs constant", "expllora_vs_matched": "Explanation LoRA vs zero-shot (384)",
              "expllora_vs_const": "Explanation LoRA vs constant", "sp_llava_vs_const": "Spatial: LLaVA vs constant",
              "sp_qwen_vs_const": "Spatial: Qwen vs constant"}
    for m in out["mcnemar"]:
        if m.get("status") == "pending":
            continue
        mc.append(f"{labels[m['test']]} & {m['a_right_b_wrong']} & {m['b_right_a_wrong']} & "
                  f"{m['p_exact_two_sided']:.3f} & {m['p_holm']:.3f} \\\\")
    (TEX_DIR / "mcnemar_rows.tex").write_text(head + "\n".join(mc) + "\n")
    write_main_tables(out, head)


def _acc_cells(r: dict) -> str:
    return (f"{pct(r['accuracy'])} ({r['k']}) & {pct(r['ci95'][0])}--{pct(r['ci95'][1])} & "
            f"{r['answered']} & {r['macro_f1']:.2f} \\\\")


def write_main_tables(out: dict, head: str) -> None:
    runs, base = out["runs"], out["baselines"]

    def rows(spec: list[tuple[str, str, str]]) -> str:
        body = []
        for key, model, inp in spec:
            r = runs.get(key) or base.get(key) or {}
            if r.get("status", "ok") != "ok" or "k" not in r:
                body.append(f"{model} & {inp} & \\multicolumn{{4}}{{l}}{{{r.get('status', 'pending')}}} \\\\")
            else:
                body.append(f"{model} & {inp} & " + _acc_cells(r))
        return head + "\n".join(body) + "\n"

    (TEX_DIR / "zeroshot_leakfree_rows.tex").write_text(rows([
        ("qwen_evid_336", "Qwen2-VL-2B", "evidence-only"),
        ("qwen_compact", "Qwen2-VL-2B", "compact"),
        ("llava_compact", "LLaVA-Interleave-0.5B", "compact"),
        ("blip_compact", "BLIP-2 OPT-2.7B", "compact, stitched"),
    ]))
    (TEX_DIR / "zeroshot_adapted_rows.tex").write_text(rows([
        ("pali_ndvi", "PaliGemma-3B", "card NDVI line only"),
        ("blip_img", "BLIP-2 OPT-2.7B$^b$", "stitched images, no card"),
    ]))
    (TEX_DIR / "zeroshot_leaky_rows.tex").write_text(rows([
        ("qwen_full", "Qwen2-VL-2B", "full card"),
        ("qwen_full_rerun", "Qwen2-VL-2B", "full card (re-run)"),
        ("llava_full", "LLaVA-Interleave-0.5B", "full card"),
    ]))
    (TEX_DIR / "zeroshot_reference_rows.tex").write_text(rows([
        ("random:const", "Best constant", "always increase"),
        ("random:train_majority", "Training majority", "always stable"),
        ("random:fixed_card", "Fixed $\\pm0.02$ rule", "card \\dndvi"),
        ("random:label_rule_card", "Eq.~(\\ref{eq:label}), batch 1", "card \\dndvi"),
        ("random:label_rule_ee", "Eq.~(\\ref{eq:label}), batch 1", "Earth Engine $\\Delta$"),
        ("random:fitted_card", "Fitted two-threshold rule", "card \\dndvi"),
        ("random:fitted_true", "Fitted two-threshold rule", "unpadded patch mean"),
    ]))

    sp = []
    for key, name in (("sp_qwen_compact", "Qwen2-VL-2B, compact"), ("sp_llava_compact", "LLaVA-Interleave-0.5B, compact"),
                      ("spatial:const", "Best constant (always stable)"), ("spatial:train_majority", "Training majority (always increase)"),
                      ("spatial:fixed_card", "Fixed $\\pm0.02$ rule, card"), ("spatial:label_rule_ee", "Eq.~(\\ref{eq:label}), Earth Engine $\\Delta$"),
                      ("spatial:fitted_card", "Rule fitted on spatial train, card"), ("spatial:fitted_true", "Rule fitted on spatial train, unpadded")):
        r = runs.get(key) or base.get(key)
        sp.append(f"{name} & {pct(r['accuracy'])} ({r['k']}) & {pct(r['ci95'][0])}--{pct(r['ci95'][1])} & {r['macro_f1']:.2f} \\\\")
    (TEX_DIR / "spatial_rows.tex").write_text(head + "\n".join(sp) + "\n")

    claims = json.loads((REVIEW / "claim_audit.json").read_text())
    ex = []
    for key, name in (("llava_freeform", "LLaVA, free-form"), ("qwen_expl_zs_336", "Qwen zero-shot, free-form"),
                      ("qwen_expl_zs_384", "Qwen zero-shot, free-form (384)"), ("expl_lora", "Explanation LoRA (384)"),
                      ("qwen_struct", "Qwen, structured"), ("qwen_struct_512", "Qwen, structured, 512 tok."),
                      ("llava_struct", "LLaVA, structured")):
        r, c = runs.get(key, {}), claims.get(key, {})
        if r.get("status") != "ok" or c.get("status") == "pending":
            ex.append(f"{name} & \\multicolumn{{6}}{{l}}{{pending}} \\\\")
            continue
        sc = c["self_consistency"]
        ex.append(f"{name} & {pct(r['accuracy'])} ({r['k']}) & {c['scored']} & "
                  f"{pct(c['consistent'] / c['scored'])} & {pct(c['consistent_excluding_rule'] / c['scored_excluding_rule'])} "
                  f"({c['consistent_excluding_rule']}/{c['scored_excluding_rule']}) & {sc[0]}/{sc[1]} & {c['unclosed_json']} \\\\")
    (TEX_DIR / "explain_rows.tex").write_text(head + "\n".join(ex) + "\n")

    sens_names = {"const_random": "Best constant", "qwen_full": "Qwen, full card", "qwen_full_rerun": "Qwen, full card (re-run)",
                  "qwen_evid_336": "Qwen, evidence-only", "qwen_compact": "Qwen, compact", "llava_compact": "LLaVA, compact",
                  "class_lora": "Class LoRA", "qwen_evid_384": "Qwen, evidence-only (384)", "qwen_struct": "Qwen, structured",
                  "llava_struct": "LLaVA, structured", "qwen_expl_zs_384": "Qwen zero-shot, free-form (384)",
                  "expl_lora": "Explanation LoRA"}
    rc, uc = out["sensitivity"]["rule_consistent"], out["sensitivity"]["unclipped"]
    sens = [f"{n} & {rc[k]['k']}/{rc[k]['n']} & {uc[k]['k']}/{uc[k]['n']} \\\\" for k, n in sens_names.items() if k in rc]
    (TEX_DIR / "sensitivity_rows.tex").write_text(head + "\n".join(sens) + "\n")


if __name__ == "__main__":
    main()
