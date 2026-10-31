#!/usr/bin/env python3
"""Workflow figure for EcoExplain-VLM report v6 (copy of make_workflow_figure_corrected.py).

v6 changes (Draft 1 review): two-stage labels, both splits, the reported balanced class LoRA,
matched zero-shot controls, spatial evaluation. Output: report/paper_v3/figures/fig_workflow_v6.png


Corrected copy made for the updated study guide. The original is
ecoexplain_vlm_dataset/papers/acl/make_workflow_figure.py; only the box texts
marked "CORRECTED" and the extra arrows marked "ADDED" differ from it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches

OUT = Path(__file__).resolve().parents[3] / "report" / "paper_v3" / "figures"
DPI = 300
W, H = 12.4, 7.35

# Palette (print-safe, colourblind-friendly)
C_IN = "#D9EDD4"
C_IN_E = "#5A8F4D"
C_PROC = "#D6E4F5"
C_PROC_E = "#3D6A99"
C_EV = "#F7E3C4"
C_EV_E = "#C4843A"
C_SPLIT = "#E4D7F2"
C_SPLIT_E = "#7A5AA3"
C_ZS = "#D4EEF0"
C_ZS_E = "#2E7A80"
C_LORA = "#F6D6D4"
C_LORA_E = "#B3544C"
C_EX = "#F3E8C8"
C_EX_E = "#9A7B2F"
INK = "#1F2430"
MUTED = "#5C6570"
TITLE = "#152033"


def box(ax, x, y, w, h, fc, ec, title, body, title_size=8.2, body_size=6.6, lw=1.15):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=0.6,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h - 0.16,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.08,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=MUTED,
        linespacing=1.35,
        zorder=3,
    )
    return (x + w / 2, y, x + w / 2, y + h)  # cx, bottom, cx, top


def arrow(ax, x1, y1, x2, y2, color="#6B7380"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.15,
            mutation_scale=11,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=1,
    )


def stage_label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=7.4, fontweight="bold", color=TITLE)


def main():
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 7.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        6.2,
        7.12,
        "EcoExplain-VLM  ·  experimental workflow",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=TITLE,
    )
    ax.text(
        6.2,
        6.82,
        "Kanha Tiger Reserve  ·  dry season T0 Jan–Feb 2022  /  T1 Jan–Feb 2023",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )

    # ----- 01 inputs -----
    stage_label(ax, 0.28, 6.48, "01  Multi-sensor inputs")
    ins = [
        (0.28, "Sentinel-2 optical", "RGB + 10 m bands\nNDVI · EVI · NDMI · NBR"),
        (2.68, "Sentinel-1 SAR", "VV / VH backscatter\ndescending pass"),
        (5.08, "CHIRPS rainfall", "Window totals\nT0 and T1"),
        (7.48, "ERA5-Land", "Temperature & soil moisture\nanomalies vs 2015–2021"),
        (9.88, "NASADEM terrain", "Elevation · slope\nstatic context"),
    ]
    tops = []
    for x, t, b in ins:
        cx, bot, _, top = box(ax, x, 5.42, 2.24, 0.92, C_IN, C_IN_E, t, b)
        tops.append((cx, bot))

    # ----- 02 gee -----
    stage_label(ax, 0.28, 5.18, "02  Preprocessing")
    gee = box(
        ax,
        0.28,
        4.22,
        11.84,
        0.82,
        C_PROC,
        C_PROC_E,
        "Earth Engine  ·  matched composites, cloud mask, 640 m patches",
        "5,800-cell grid → 3,668 forested cells (WorldCover) → 1,200 scored candidates   ·   15 GeoTIFFs per patch   ·   T0/T1 RGB chips",
        title_size=8.4,
        body_size=7.0,
    )
    for cx, bot in tops:
        arrow(ax, cx, bot - 0.02, cx, gee[3] + 0.02)

    # ----- 03 evidence -----
    stage_label(ax, 0.28, 3.98, "03  Labels and evidence")
    b_rgb = box(
        ax,
        0.28,
        2.88,
        3.72,
        0.96,
        C_EV,
        C_EV_E,
        "Paired RGB chips",
        # CORRECTED: say where the chips go and that BLIP-2 gets them stitched
        "True-colour T0 and T1 on a 512 × 512 canvas\nimage input (BLIP-2: one stitched image)",
    )
    b_lab = box(
        ax,
        4.34,
        2.88,
        3.72,
        0.96,
        C_EV,
        C_EV_E,
        "Automatic class  (gold)",
        # CORRECTED: give the actual cut-offs
        "Relative ΔNDVI class, two batches:\nbatch 1 (296): pool percentiles −0.038 / +0.003\nbatch 2 (70): recomputed −0.034 / −0.001",
    )
    b_card = box(
        ax,
        8.40,
        2.88,
        3.72,
        0.96,
        C_EV,
        C_EV_E,
        "Structured evidence card",
        # CORRECTED: say exactly where the label sits
        "Indices, SAR, climate, terrain\nsummary line prints trend word + label\n(removed in every leak-free input)",
    )
    arrow(ax, 6.2, gee[1] - 0.02, 6.2, b_lab[3] + 0.02)
    # ADDED: Earth Engine feeds all three boxes, not only the label
    # (left arrow at x = 3.0 so it does not run through the "03" stage label)
    arrow(ax, 3.0, gee[1] - 0.02, 3.0, b_rgb[3] + 0.02)
    arrow(ax, 10.26, gee[1] - 0.02, 10.26, b_card[3] + 0.02)

    # ----- 04 split -----
    stage_label(ax, 0.28, 2.64, "04  Dataset freeze")
    b_split = box(
        ax,
        0.28,
        1.78,
        11.84,
        0.72,
        C_SPLIT,
        C_SPLIT_E,
        "366 patches  →  random split 256 / 55 / 55   ·   spatial block split 266 / 46 / 54",
        "Gold = candidate_class (frozen)   ·   best constant on random test = 34.5%   ·   protocols compared on the same test patches within each split",
        title_size=8.4,
        body_size=7.0,
    )
    arrow(ax, 6.2, b_lab[1] - 0.02, 6.2, b_split[3] + 0.02)
    # ADDED: images and cards also go into every record
    arrow(ax, 3.0, b_rgb[1] - 0.02, 3.0, b_split[3] + 0.02)
    arrow(ax, 10.26, b_card[1] - 0.02, 10.26, b_split[3] + 0.02)

    # ----- 05 three tracks -----
    # CORRECTED: state why the scores must not be mixed
    stage_label(ax, 0.28, 1.54, "05  Three evaluation tracks + spatial check  (compare only within a protocol)")
    b_zs = box(
        ax,
        0.28,
        0.22,
        3.72,
        1.18,
        C_ZS,
        C_ZS_E,
        "A  Zero-shot classification",
        # CORRECTED: the four models did not receive the same input
        "full card (leaky) vs evidence-only (same\nimages, summary line removed) vs compact\ntable (identical for every model)\nspatial test: compact table",
        body_size=6.5,
    )
    b_cl = box(
        ax,
        4.34,
        0.22,
        3.72,
        1.18,
        C_LORA,
        C_LORA_E,
        # CORRECTED: describe the reported run by its size, and why the 100% runs are excluded
        "B  Class LoRA  (16 balanced patches)",
        "Qwen2-VL-2B · r = 8 · 6/5/5 · 1 epoch\nevidence-only at train and test, 384 px\nvs matched zero-shot control\nfull-card runs: shortcut, appendix only",
        body_size=6.5,
    )
    b_ex = box(
        ax,
        8.40,
        0.22,
        3.72,
        1.18,
        C_EX,
        C_EX_E,
        "C  Explanations  (label removed)",
        # CORRECTED: name the three phases and what is scored
        "free-form vs structured cite-only prompt\ndirectional claim consistency vs card Δ\nexplanation LoRA (16 decline targets)\nvs matched zero-shot control",
        body_size=6.5,
    )
    arrow(ax, 2.14, b_split[1] - 0.02, 2.14, b_zs[3] + 0.02)
    arrow(ax, 6.20, b_split[1] - 0.02, 6.20, b_cl[3] + 0.02)
    arrow(ax, 10.26, b_split[1] - 0.02, 10.26, b_ex[3] + 0.02)

    png = OUT / "fig_workflow_v6.png"
    pdf = OUT / "fig_workflow_v6.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
