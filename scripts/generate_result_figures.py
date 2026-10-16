#!/usr/bin/env python3
"""Generate result1.png (flowchart) and result2.png (results table)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT1 = ROOT / "results" / "result1.png"
OUT2 = ROOT / "results" / "result2.png"
OUT_DETAIL = ROOT / "results" / "ecoexplain_vlm_pipeline.png"
COMP = ROOT / "results" / "vlm_combo_comparison.json"

# Language side actually used at eval (LAPTOP=True checkpoints); not paper 7B names.
TEXT_ENCODER_USED: dict[str, str] = {
    "qwen_vl_custom_mlp_qwen": "Qwen2 tokenizer + Qwen2 LM (2B Instruct)",
    "paligemma_siglip_linear_gemma": "Gemma tokenizer + Gemma LM (3B mix-224)",
    "llava15_clip_mlp_vicuna": "Qwen2 tokenizer + LM (0.5B, LLaVA-Interleave)",
    "blip2_clip_qformer_opt": "OPT tokenizer + OPT LM (2.7B)",
}


def draw_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")

    def box(x, y, w, h, text, fc="#e8f4fc", ec="#1a5276", fontsize=9, bold=False):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        weight = "bold" if bold else "normal"
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight=weight,
            wrap=True,
            multialignment="center",
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#444444",
            )
        )

    ax.text(
        7,
        9.55,
        "EcoExplain-VLM — Kanha Tiger Reserve (Jan–Feb 2022 vs 2023)",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color="#1a1a1a",
    )
    ax.text(
        7,
        9.15,
        "Condition C: spectral evidence text + T0/T1 RGB  →  stable | decline | increase",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )

    # Row 1 — data
    box(0.4, 7.35, 2.6, 1.15, "01 GEE export\nS2 · SAR · CHIRPS\nERA5 · NASADEM", fc="#d5e8d4")
    box(3.35, 7.35, 2.6, 1.15, "02 DIP + RGB\nOtsu masks\nterrain previews", fc="#d5e8d4")
    box(6.3, 7.35, 2.8, 1.15, "03 JSONL v2\nunique Condition C\nprompts per patch", fc="#d5e8d4")
    box(9.55, 7.35, 3.9, 1.15, "04 Freeze splits\n366 patches\n256 train · 55 dev · 55 test", fc="#fff2cc", bold=True)

    arrow(3.0, 7.92, 3.35, 7.92)
    arrow(5.95, 7.92, 6.3, 7.92)
    arrow(9.1, 7.92, 9.55, 7.92)

    # Dataset hub
    box(4.2, 5.85, 5.6, 1.0, "dataset.jsonl + data/processed_images/*.jpg\n(index only on GitHub; rasters local)", fc="#f5f5f5", ec="#888888")

    arrow(7, 7.35, 7, 6.85)

    # Row 2 — eval
    box(0.5, 4.15, 2.5, 1.2, "05 LLaVA-1.5\n(0.5B laptop)", fc="#dae8fc")
    box(3.35, 4.15, 2.5, 1.2, "06 PaliGemma\nSigLIP + Gemma", fc="#dae8fc")
    box(6.2, 4.15, 2.5, 1.2, "07 Qwen2-VL\n2B instruct", fc="#dae8fc")
    box(9.05, 4.15, 2.5, 1.2, "08 BLIP-2\nQ-Former + OPT", fc="#dae8fc")

    for x in (1.75, 4.6, 7.45, 10.3):
        arrow(x, 5.85, x, 5.35)

    # Merge
    box(3.8, 2.65, 6.4, 1.05, "Same 55 test patches · shared vlm_eval.py\nresults/vlm_combo_comparison.json", fc="#e1d5e7", bold=True)

    for x in (1.75, 4.6, 7.45, 10.3):
        arrow(x, 4.15, 7, 3.7)

    # 09 optional
    box(4.5, 0.55, 5.0, 1.05, "09 LoRA fine-tune Qwen (GPU)\nqwen_vl_finetuned_ecoexplain", fc="#ffffff", ec="#999999", fontsize=8.5)
    ax.plot([7, 7], [2.65, 1.6], linestyle="--", color="#999999", linewidth=1)
    arrow(7, 1.6, 7, 1.6)

    ax.text(
        7,
        0.25,
        "GitHub: Rama-Vaibhav/EcoExplain-VLM  ·  zero-shot laptop checkpoints (LAPTOP=True)",
        ha="center",
        fontsize=8,
        color="#666666",
    )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.35, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_detailed_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 12), dpi=220)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f8f6")
    ax.set_facecolor("#f7f8f6")

    palette = {
        "source": ("#e4f0e6", "#427753"),
        "prep": ("#e5eef5", "#356b8c"),
        "evidence": ("#f8edda", "#a46d22"),
        "dataset": ("#e9e7f2", "#65588b"),
        "eval": ("#e1eef0", "#34747b"),
        "output": ("#f0e7e4", "#9a5a48"),
    }

    def node(x, y, w, h, title, body, kind, title_size=10.5, body_size=8.0):
        fill, edge = palette[kind]
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.035,rounding_size=0.10",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.16, y + h - 0.22, title,
            ha="left", va="top", fontsize=title_size, weight="bold", color="#20313a",
        )
        ax.text(
            x + 0.16, y + h - 0.52, body,
            ha="left", va="top", fontsize=body_size, color="#34434a",
            linespacing=1.35,
        )

    def arrow(x1, y1, x2, y2, color="#5d6b70", style="solid", width=1.25):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=13, linewidth=width,
            linestyle=style, color=color, connectionstyle="arc3,rad=0.0",
        ))

    ax.text(9, 11.62, "EcoExplain-VLM | End-to-End Research Pipeline",
            ha="center", va="center", fontsize=19, weight="bold", color="#1d3038")
    ax.text(9, 11.24,
            "Kanha Tiger Reserve  ·  matched dry-season windows: Jan–Feb 2022 (T0) and Jan–Feb 2023 (T1)",
            ha="center", va="center", fontsize=10.5, color="#53636a")

    ax.text(0.65, 10.85, "01  MULTIMODAL INPUTS", fontsize=8.5, weight="bold", color="#637178")
    sources = [
        ("Sentinel-2 optical", "RGB + 10 bands\nNDVI · EVI · NDMI · NBR"),
        ("Sentinel-1 SAR", "VV / VH backscatter\ndominant orbit pass"),
        ("CHIRPS rainfall", "Jan–Feb rainfall totals\npatch-level context"),
        ("ERA5-Land", "temperature + soil-moisture\nanomalies vs 2015–2021"),
        ("NASADEM terrain", "elevation · slope · aspect\nstatic terrain context"),
    ]
    source_x = [0.65, 4.02, 7.39, 10.76, 14.13]
    for x, (title, body) in zip(source_x, sources):
        node(x, 9.42, 3.05, 1.18, title, body, "source", title_size=9.3, body_size=7.25)

    ax.text(0.65, 9.12, "02  PREPROCESSING & SPATIAL ALIGNMENT", fontsize=8.5, weight="bold", color="#637178")
    node(
        0.65, 7.92, 16.53, 0.92,
        "Earth Engine export and patch harmonization",
        "Kanha AOI + matched date filters  →  cloud/quality masking and temporal composites  →  common patch geometry (64 × 64 pixels at 10 m)\n"
        "Export paired T0/T1 rasters; retain native scale for coarse climate products and terrain context.",
        "prep", title_size=10.2, body_size=8.0,
    )
    for x in [2.175, 5.545, 8.915, 12.285, 15.655]:
        arrow(x, 9.42, x, 8.88, width=1.0)

    ax.text(0.65, 7.58, "03  PATCH-LEVEL CHANGE & EVIDENCE", fontsize=8.5, weight="bold", color="#637178")
    node(
        0.65, 5.92, 5.15, 1.40,
        "Optical change features",
        "Compute T1 − T0 for NDVI and NBR.\nUse index stacks and paired RGB chips\nto describe vegetation change.",
        "prep", title_size=9.6, body_size=7.8,
    )
    node(
        6.43, 5.92, 5.15, 1.40,
        "DIP / Otsu disturbance extraction",
        "Threshold negative ΔNDVI / ΔNBR signals.\nCreate masks, contours, bounding boxes,\nand severity triplets; inspect QA panels.",
        "evidence", title_size=9.6, body_size=7.8,
    )
    node(
        12.20, 5.92, 5.15, 1.40,
        "Aligned multimodal patch stack",
        "Pad / resize imagery, masks, and sensor tensors\nto the 512 × 512 VLM canvas; transform\nbounding boxes consistently.",
        "prep", title_size=9.6, body_size=7.8,
    )
    arrow(6.10, 7.92, 3.23, 7.32)
    arrow(9.0, 7.92, 9.0, 7.32)
    arrow(11.90, 7.92, 14.78, 7.32)
    arrow(5.80, 6.62, 6.43, 6.62)
    arrow(12.20, 6.62, 11.58, 6.62)

    ax.text(0.65, 5.58, "04  RECORD ASSEMBLY", fontsize=8.5, weight="bold", color="#637178")
    node(
        2.0, 4.15, 14.0, 1.15,
        "Evidence card + structured JSONL record",
        "T0/T1 RGB · Sentinel-2 optical/indices · Sentinel-1 · rainfall · ERA5-Land · DEM · ΔNDVI/ΔNBR rasters and masks · disturbance triplets\n"
        "Attach candidate_class and reference_class separately; include measured context and explanation text. Strip coordinates from VLM-facing records.",
        "evidence", title_size=10.2, body_size=8.0,
    )
    arrow(9.0, 5.92, 9.0, 5.30)

    ax.text(0.65, 3.81, "05  DATASET FREEZE & EVALUATION", fontsize=8.5, weight="bold", color="#637178")
    node(
        2.0, 2.48, 14.0, 1.05,
        "Freeze dataset and stratified splits (seed 42)",
        "366 patch records  ·  Train 256 (LoRA training)  ·  Dev 55 (development)  ·  Test 55 (held-out benchmark)",
        "dataset", title_size=10.2, body_size=8.2,
    )
    arrow(9.0, 4.15, 9.0, 3.53)

    node(
        0.65, 0.64, 7.65, 1.35,
        "Zero-shot Condition C benchmark",
        "Same held-out test patches; paired RGB + evidence text.\nLLaVA-1.5 · PaliGemma · Qwen2-VL · BLIP-2\nParse stable / decline / increase; compare accuracy.",
        "eval", title_size=9.5, body_size=7.6,
    )
    node(
        9.70, 0.64, 7.65, 1.35,
        "Qwen2-VL LoRA fine-tune + test",
        "Train on the train split; evaluate on the same held-out test split.\nSave fine-tuned predictions and comparison\nartifacts alongside zero-shot results.",
        "output", title_size=9.5, body_size=7.6,
    )
    arrow(5.0, 2.48, 4.5, 1.99)
    arrow(13.0, 2.48, 13.5, 1.99)

    ax.text(9, 0.25,
            "Outputs: frozen split manifest · per-model predictions / result JSON · merged comparison report  |  Explanation quality still requires independent validation",
            ha="center", va="center", fontsize=8.1, color="#53636a")

    fig.savefig(path, bbox_inches="tight", pad_inches=0.24, facecolor=fig.get_facecolor())
    plt.close(fig)


def _wrap_lines(text: str, max_chars: int) -> str:
    """Break text into lines so each line fits within max_chars (word-aware)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    words = text.replace("/", " / ").split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for w in words:
        add = len(w) + (1 if current else 0)
        if current and length + add > max_chars:
            lines.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
            length += add
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _style_table(tbl, header_rows: int = 1, highlight_row: int | None = None) -> None:
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.8)
        cell.PAD = 0.08
        if row < header_rows:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", weight="bold", fontsize=9)
            cell.set_height(0.12)
        elif highlight_row is not None and row == highlight_row:
            cell.set_facecolor("#d4edda")
            cell.set_text_props(fontsize=8.5)
            cell.set_height(0.14)
        else:
            cell.set_facecolor("#f4f4f4" if row % 2 == 0 else "#ffffff")
            cell.set_text_props(fontsize=8.5)
            cell.set_height(0.14)


def draw_results_table(path: Path) -> None:
    data = json.loads(COMP.read_text())
    rows = sorted(data["table"], key=lambda r: r.get("accuracy") or 0, reverse=True)

    fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(
        0.5,
        0.97,
        "EcoExplain-VLM — Zero-shot benchmark results",
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
    )
    ax.text(
        0.5,
        0.925,
        f"Test split · {data['max_eval']}/55 patches · 3-class change: stable / decline / increase",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    ax.text(
        0.5,
        0.895,
        "Input: T0 + T1 JPG and Condition C text (each VLM's own tokenizer; no separate text encoder)",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )
    ax.text(
        0.5,
        0.868,
        "Metric: classification accuracy (parsed label vs candidate_class)",
        ha="center",
        va="top",
        fontsize=9,
        color="#666666",
    )

    col_labels = ["Rank", "Model", "Vision encoder", "Text encoder used", "Accuracy", "N", "Status"]
    col_widths = [0.06, 0.11, 0.20, 0.32, 0.10, 0.07, 0.08]

    table_data = []
    for i, r in enumerate(rows, start=1):
        cid = r.get("combo_id", "")
        text_enc = TEXT_ENCODER_USED.get(cid) or _wrap_lines(r.get("llm", "—"), 24)
        table_data.append(
            [
                str(i),
                _wrap_lines(r["family"], 12),
                _wrap_lines(r["vision"], 22),
                _wrap_lines(text_enc, 38),
                r.get("accuracy_pct", "—"),
                str(r.get("n_eval", "")),
                r.get("status", ""),
            ]
        )

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
        bbox=[0.04, 0.46, 0.92, 0.36],
        colWidths=col_widths,
    )
    _style_table(tbl, highlight_row=1)

    summary_labels = ["Dataset", "Splits (seed 42)", "Best zero-shot", "Checkpoints", "Artifacts"]
    summary_values = [
        "366 patches · Kanha Tiger Reserve · Jan–Feb dry season",
        "Train 256  ·  Dev 55  ·  Test 55",
        f"{data['best_combo_id']}  ({data['best_accuracy_pct']})",
        "Laptop HF sizes: Qwen 2B, LLaVA 0.5B, PaliGemma 3B, BLIP-2 2.7B",
        "Per-model JSON under results/vlm_combo_*.json; merged vlm_combo_comparison.json",
    ]
    summary_data = [
        [_wrap_lines(summary_labels[i], 18), _wrap_lines(summary_values[i], 72)]
        for i in range(len(summary_labels))
    ]

    ax.text(0.04, 0.44, "Context", fontsize=11, weight="bold", color="#333333", va="top")

    st = ax.table(
        cellText=summary_data,
        colLabels=["Item", "Details"],
        cellLoc="left",
        loc="upper center",
        bbox=[0.04, 0.08, 0.92, 0.34],
        colWidths=[0.22, 0.78],
    )
    st.auto_set_font_size(False)
    for (row, col), cell in st.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.PAD = 0.06
        if row == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold", fontsize=9)
            cell.set_height(0.06)
        elif col == 0:
            cell.set_facecolor("#f5f5f5")
            cell.set_text_props(weight="bold", fontsize=8.5)
            cell.set_height(0.075)
        else:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(fontsize=8.5)
            cell.set_height(0.075)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT1.parent.mkdir(parents=True, exist_ok=True)
    draw_flowchart(OUT1)
    draw_results_table(OUT2)
    draw_detailed_pipeline(OUT_DETAIL)
    print("wrote", OUT1)
    print("wrote", OUT2)
    print("wrote", OUT_DETAIL)


if __name__ == "__main__":
    main()
