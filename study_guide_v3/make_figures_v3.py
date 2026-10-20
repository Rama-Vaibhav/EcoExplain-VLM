"""Figures for the EcoExplain-VLM study guide, version 3 (20 October 2026).

Every number is read from a project file. Run from anywhere:

    python make_figures_v3.py

It (1) copies the figures you already have into ./figures and
(2) draws:
    fig_accuracy_v3.png      every classification run, labelled by what the model was given,
                             including the leak-free reruns of 19-20 October
    fig_lora_v3.png          training-class mix and test confusion of all three LoRA runs
    fig_leakfree_confusion.png  Qwen with and without the summary line, and LLaVA (compact)
    fig_explain_phases.png   explanation scores for Phases B, C and D side by side
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
# Works both beside the repo (local layout) and inside it (GitHub copy).
REPO = TOP if (TOP / "results").is_dir() else TOP / "ecoexplain_vlm_dataset"
ADA = TOP / "ada_review" / "EcoExplain-VLM-results"
OLD = TOP / "study_guide" / "figures"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

CLASSES = ["decline", "stable", "increase"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def preds(path):
    return json.loads(Path(path).read_text())["predictions"]


rows = {json.loads(l)["id"]: json.loads(l) for l in (REPO / "dataset.jsonl").open()}
split = {
    s: [r["patch_id"] for r in csv.DictReader((REPO / f"data/splits/{s}_patches.csv").open())]
    for s in ("train", "dev", "test")
}

# 1. Copy the figures that already exist ------------------------------------------
copies = [
    (REPO / "papers/acl/fig_workflow.png", "fig_workflow_yours.png"),
    (REPO / "results/result1.png", "fig_result1_old.png"),
    (REPO / "results/result2.png", "fig_result2_old.png"),
    (REPO / "results/explain/summary_scores.png", "fig_summary_scores_old.png"),
]
for name in [
    "fig_pipeline.png", "fig_patch_map.png", "fig_ndvi_labels.png", "fig_label_vs_trend.png",
    "fig_examples.png", "fig_padding_bias.png", "fig_climate_by_class.png",
    "fig_accuracy_overview.png", "fig_confusion.png", "fig_explain_structured.png",
    "fig_panel_p00575.png", "fig_otsu_p01289.jpg", "fig_purple.jpg", "fig_natural.jpg",
]:
    copies.append((OLD / name, name))
for src, dst in copies:
    if src.exists():
        shutil.copy2(src, OUT / dst)
        print("copied", dst)
    else:
        print("MISSING", src)

# 2. Accuracy of every classification run, labelled by input ----------------------


def ndvi_rule(pid, lo, hi):
    d = rows[pid]["file_analysis"]["index_delta"]["NDVI"]
    return "decline" if d <= lo else ("increase" if d >= hi else "stable")


test = split["test"]
k_fixed = sum(ndvi_rule(i, -0.02, 0.02) == rows[i]["candidate_class"] for i in test)

vals = sorted({round(rows[i]["file_analysis"]["index_delta"]["NDVI"], 4) for i in split["train"]})
best = None
for lo_t in [v for v in vals if v < 0.01]:
    for hi_t in [v for v in vals if v > lo_t]:
        a = sum(ndvi_rule(i, lo_t, hi_t) == rows[i]["candidate_class"] for i in split["train"])
        if best is None or a > best[0]:
            best = (a, lo_t, hi_t)
k_fit = sum(ndvi_rule(i, best[1], best[2]) == rows[i]["candidate_class"] for i in test)
print(f"rule +-0.02: {k_fixed}/55 | train-fitted rule {best[1]}, {best[2]}: {k_fit}/55")

PROT = REPO / "results/protocol"
runs = [
    ("Qwen2-VL-2B zero-shot", "images + full card", REPO / "results/vlm_combo_qwen_vl_custom_mlp_qwen.json", "full"),
    ("Qwen2-VL-2B class LoRA, first 16 by ID", "full card (superseded)", REPO / "results/vlm_combo_qwen_vl_finetuned_ecoexplain.json", "full"),
    ("LLaVA-Interleave-0.5B zero-shot", "images + full card", REPO / "results/vlm_combo_llava15_clip_mlp_vicuna.json", "full"),
    ("PaliGemma-3B zero-shot", "images + NDVI line only", REPO / "results/vlm_combo_paligemma_siglip_linear_gemma.json", "short"),
    ("BLIP-2 OPT-2.7B zero-shot", "stitched image, no card (prompt echo)", REPO / "results/vlm_combo_blip2_clip_qformer_opt.json", "short"),
    ("Qwen2-VL-2B zero-shot (20 Oct)", "evidence_only card, label line removed", None, "free"),
    ("Qwen2-VL-2B zero-shot (20 Oct)", "compact table, same for all models", PROT / "vlm_combo_qwen_vl_custom_mlp_qwen.json", "free"),
    ("LLaVA-Interleave-0.5B zero-shot (20 Oct)", "compact table, same for all models", PROT / "vlm_combo_llava15_clip_mlp_vicuna.json", "free"),
    ("Qwen2-VL-2B class LoRA, balanced 16 (Ada 29575)", "evidence_only at train and test", PROT / "qwen_balanced_leakfree_lora.json", "free"),
    ("LLaVA evidence_only (2 Oct)", "card without the label line", REPO / "results/explain/oneword_llava15_clip_mlp_vicuna.json", "clean"),
    ("Qwen LoRA, Ada job 3054", "trained with card, tested on images only", ADA / "vlm_combo_qwen_vl_finetuned_ecoexplain.json", "probe"),
    ("LLaVA closed-set probe", "class-word scores + dev-fitted bias", ADA / "vlm_closedset_llava15_clip_mlp_vicuna.json", "probe"),
]
labels, accs, lo_err, hi_err, kinds = [], [], [], [], []
for name, inp, path, kind in runs:
    if path is None:  # Qwen evidence_only: only the comparison summary survives
        ev = json.loads((PROT / "vlm_combo_comparison_evidence_only.json").read_text())["ranking"][0]
        n = ev["n_eval"]
        k = round(ev["accuracy"] * n)
    else:
        p = preds(path)
        k = sum(1 for x in p if x.get("pred") == x["gold"])
        n = len(p)
    lo, hi = wilson(k, n)
    labels.append(f"{name}\n[{inp}]")
    accs.append(k / n)
    lo_err.append(k / n - lo)
    hi_err.append(hi - k / n)
    kinds.append(kind)
    print(f"{name}: {k}/{n} = {100 * k / n:.1f}%")
for name, inp, k in [
    ("ΔNDVI rule, fixed ±0.02 cut-offs", "no model; card ΔNDVI", k_fixed),
    ("ΔNDVI rule, cut-offs fitted on train", "no model; card ΔNDVI", k_fit),
]:
    lo, hi = wilson(k, 55)
    labels.append(f"{name}\n[{inp}]")
    accs.append(k / 55)
    lo_err.append(k / 55 - lo)
    hi_err.append(hi - k / 55)
    kinds.append("rule")

kc = {"full": "#e67e22", "short": "#95a5a6", "free": "#1f4e79", "clean": "#5dade2", "probe": "#8e44ad", "rule": "#16a085"}
fig, ax = plt.subplots(figsize=(10.5, 9.6))
y = np.arange(len(labels))[::-1]
ax.barh(y, accs, xerr=[lo_err, hi_err], color=[kc[k] for k in kinds], capsize=3, height=0.72)
for yi, a, h in zip(y, accs, hi_err):
    ax.text(a + h + 0.012, yi, f"{100 * a:.1f}%", fontsize=9, va="center")
ax.axvline(19 / 55, color="black", ls="--", lw=1)
ax.text(19 / 55 + 0.006, len(labels) - 0.35, "majority baseline 34.5%", fontsize=8.5)
ax.set_yticks(y, labels, fontsize=8.3)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Test accuracy on the same 55 patches, with 95% Wilson interval")
ax.legend(
    handles=[
        Patch(color=kc["full"], label="Condition C with the full card (contains candidate_class)"),
        Patch(color=kc["short"], label="Condition C with a short input (no summary line)"),
        Patch(color=kc["free"], label="leak-free reruns of 19-20 Oct (no summary line)"),
        Patch(color=kc["clean"], label="earlier evidence_only re-run (2 Oct)"),
        Patch(color=kc["probe"], label="exploratory run or probe (not zero-shot generation)"),
        Patch(color=kc["rule"], label="no model: threshold on the card's ΔNDVI"),
    ],
    fontsize=8.2, loc="upper center", bbox_to_anchor=(0.32, -0.08), ncol=2, frameon=False,
)
save(fig, "fig_accuracy_v3.png")

# 3. The two LoRA runs: what they were trained on and how they answered -----------
train_sorted = split["train"]  # train_patches.csv is sorted by id, the same order the trainer uses
class_lora_train = Counter(rows[i]["candidate_class"] for i in train_sorted[:16])
targets = [json.loads(l) for l in (REPO / "data/explain_targets/train_targets.jsonl").open()]
first128 = set(train_sorted[:128])
used = [t for t in targets if t["id"] in first128]
expl_lora_train = Counter(t["change_class"] for t in used)
print("class LoRA train mix:", dict(class_lora_train), "| explanation LoRA train mix:", dict(expl_lora_train),
      f"({len(used)} of {len(targets)} targets)")


def confusion(path):
    m = np.zeros((3, 3), int)
    for x in preds(path):
        if x.get("pred") in CLASSES:
            m[CLASSES.index(x["gold"]), CLASSES.index(x["pred"])] += 1
    return m


cm_class = confusion(REPO / "results/vlm_combo_qwen_vl_finetuned_ecoexplain.json")
cm_expl = confusion(REPO / "results/explain/explain_qwen_vl_finetuned_ecoexplain.json")
cm_bal = confusion(REPO / "results/protocol/qwen_balanced_leakfree_lora.json")
# Ada 29575 log: "train class mix: {'stable': 5, 'decline': 6, 'increase': 5}"
log = (REPO / "results/slurm-qwen-balanced-smoke-29575.log").read_text()
import ast, re
bal_train = Counter(ast.literal_eval(re.search(r"train class mix: (\{.*?\})", log).group(1)))
print("balanced LoRA train mix:", dict(bal_train))

fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1, 1, 1]})
ax = axes[0]
x = np.arange(3)
col = {"decline": "#c0392b", "stable": "#7f8c8d", "increase": "#27ae60"}
for off, mix, hatch, lab in [(-0.27, class_lora_train, "", "class LoRA, first 16 (2722446)"),
                              (0.0, bal_train, "..", "class LoRA, balanced (29575)"),
                              (0.27, expl_lora_train, "//", "explanation LoRA (24389)")]:
    ax.bar(x + off, [mix[c] for c in CLASSES], 0.25, color=[col[c] for c in CLASSES], edgecolor="black",
           hatch=hatch, label=lab)
    for xi, c in zip(x, CLASSES):
        ax.text(xi + off, mix[c] + 0.3, mix[c], ha="center", fontsize=9)
ax.set_xticks(x, CLASSES)
ax.set_ylim(0, 21)
ax.set_ylabel("training examples")
ax.set_title("(a) Classes of the 16 training examples", fontsize=10.5)
ax.legend(fontsize=7.8, frameon=False, loc="upper right")
for ax, m, title in [
    (axes[1], cm_class, "(b) Class LoRA, first 16 by ID: 67.3%\nfull card (superseded)"),
    (axes[2], cm_bal, "(c) Class LoRA, balanced 16: 36.4%\nlabel-free card (Ada 29575)"),
    (axes[3], cm_expl, "(d) Explanation LoRA: 61.8%\nlabel-free table, JSON answer"),
]:
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=19)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3), CLASSES, fontsize=9)
    ax.set_yticks(range(3), CLASSES, fontsize=9)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("gold (candidate_class)", fontsize=9)
    ax.set_title(title, fontsize=10.5)
fig.tight_layout()
save(fig, "fig_lora_v3.png")

# 3b. Zero-shot with and without the summary line ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
for ax, path, title in [
    (axes[0], REPO / "results/vlm_combo_qwen_vl_custom_mlp_qwen.json", "(a) Qwen2-VL-2B, full card: 56.4%"),
    (axes[1], REPO / "results/protocol/vlm_combo_qwen_vl_custom_mlp_qwen.json", "(b) Qwen2-VL-2B, compact table: 32.7%"),
    (axes[2], REPO / "results/protocol/vlm_combo_llava15_clip_mlp_vicuna.json", "(c) LLaVA-0.5B, compact table: 43.6%"),
]:
    m = confusion(path)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=19)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=12, color="white" if m[i, j] > 12 else "black")
    ax.set_xticks(range(3), CLASSES, fontsize=9)
    ax.set_yticks(range(3), CLASSES, fontsize=9)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("gold (candidate_class)", fontsize=9)
    ax.set_title(title, fontsize=10.5)
fig.tight_layout()
save(fig, "fig_leakfree_confusion.png")


# 4. Explanation scores across Phases B, C and D -----------------------------------
# Phase B (legacy prompt): scripts/score_explanations.py --include-legacy, as recorded in the first guide.
phase_b = {
    "Qwen2-VL-2B\nfree-form (n=3)": {"acc": 1.0, "claims": 7 / 16, "wrong": 1.0, "self": 0.667},
    "LLaVA-0.5B\nfree-form": {"acc": 0.291, "claims": 34 / 60, "wrong": 19 / 55, "self": 0.103},
}
# Phase C (structured prompt, 6 Oct): results/explain/explanation_scores.csv
csvrows = {r["combo_id"]: r for r in csv.DictReader((REPO / "results/explain/explanation_scores.csv").open())}
phase_c = {}
for cid, name in [("qwen_vl_custom_mlp_qwen", "Qwen2-VL-2B\nstructured"), ("llava15_clip_mlp_vicuna", "LLaVA-0.5B\nstructured")]:
    r = csvrows[cid]
    assert r["prompt_style"] == "structured", r
    phase_c[name] = {"acc": float(r["accuracy"]), "claims": float(r["pct_correct_claims"]),
                     "wrong": float(r["hallucination_rate"]), "self": float(r["self_consistency"])}
# Phase D (explanation LoRA, 17 Oct): results/explain/explanation_scores.json
jrows = {r["combo_id"]: r for r in json.loads((REPO / "results/explain/explanation_scores.json").read_text())["models"]}
r = jrows["qwen_vl_finetuned_ecoexplain"]
phase_d = {"Qwen2-VL-2B\nexplanation LoRA": {"acc": r["accuracy"], "claims": r["pct_correct_claims"],
                                             "wrong": r["hallucination_rate"], "self": r["self_consistency"]}}
allruns = {**phase_b, **phase_c, **phase_d}
for k, v in allruns.items():
    print(k.replace("\n", " "), {m: round(100 * x, 1) for m, x in v.items()})

metrics = [("acc", "Class accuracy\n(explanation mode)"), ("claims", "Correct claims"),
           ("wrong", "Explanations with\n≥1 wrong claim"), ("self", "Self-consistency")]
colors = ["#c8ccd0", "#9aa3ab", "#2471a3", "#5dade2", "#ca6f1e"]
hatches = ["", "", "", "", "//"]
fig, ax = plt.subplots(figsize=(12.5, 4.6))
nrun = len(allruns)
w = 0.16
xm = np.arange(len(metrics))
for i, (name, v) in enumerate(allruns.items()):
    xs = xm + (i - (nrun - 1) / 2) * w
    vals = [v[m] for m, _ in metrics]
    ax.bar(xs, vals, w * 0.95, color=colors[i], hatch=hatches[i], edgecolor="#333333", linewidth=0.5,
           label=name.replace("\n", " "))
    for xi, val in zip(xs, vals):
        ax.text(xi, val + 0.015, f"{100 * val:.0f}", ha="center", fontsize=7.5)
ax.set_xticks(xm, [lab for _, lab in metrics], fontsize=9.5)
ax.set_ylim(0, 1.18)
ax.set_ylabel("share (numbers above bars in %)")
ax.legend(fontsize=8.3, ncol=5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.13))
ax.text(0.0, -0.22, "Grey: Phase B (free-form prompt). Blue: Phase C (structured cite-only prompt). "
        "Orange: Phase D (explanation LoRA). Qwen's free-form run covers only 3 patches.",
        transform=ax.transAxes, fontsize=8.5, color="#444444")
save(fig, "fig_explain_phases.png")
