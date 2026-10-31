#!/usr/bin/env python3
"""Explanation-run figure for report v6 (replaces fig_explain_phases.png).

Reads results/review/stats_report.json and results/review/claim_audit.json (run those first).
    .venv/bin/python scripts/review/make_explain_figure.py
Output: report/paper_v3/figures/fig_explain_v6.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REVIEW, ROOT  # noqa: E402

OUT = ROOT.parent / "report" / "paper_v3" / "figures" / "fig_explain_v6.png"
SETTINGS = [
    ("llava_freeform", "LLaVA\nfree-form", "#9aa0a6"),
    ("qwen_expl_zs_384", "Qwen zero-shot\nfree-form (384)", "#6c757d"),
    ("expl_lora", "Explanation\nLoRA (384)", "#c8691c"),
    ("qwen_struct", "Qwen\nstructured", "#2171b5"),
    ("llava_struct", "LLaVA\nstructured", "#9ecae1"),
]


def main() -> None:
    stats = json.loads((REVIEW / "stats_report.json").read_text())["runs"]
    claims = json.loads((REVIEW / "claim_audit.json").read_text())
    shown = [(k, lab, col) for k, lab, col in SETTINGS
             if stats.get(k, {}).get("status") == "ok" and claims.get(k, {}).get("status") != "pending"]
    metrics = [
        ("Class accuracy", lambda r, c: (r["k"], r["n"])),
        ("Claim consistency\n(all scored claims)", lambda r, c: (c["consistent"], c["scored"])),
        ("Claim consistency\n(no copied rule text)", lambda r, c: (c["consistent_excluding_rule"], c["scored_excluding_rule"])),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    width = 0.8 / len(shown)
    x = np.arange(len(metrics))
    for i, (k, lab, col) in enumerate(shown):
        vals = [f(stats[k], claims[k]) for _, f in metrics]
        h = [a / b for a, b in vals]
        bars = ax.bar(x + (i - (len(shown) - 1) / 2) * width, h, width, color=col, edgecolor="#333333", lw=0.5,
                      label=lab.replace("\n", " "))
        for b, (a, n) in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{a}/{n}", ha="center", va="bottom",
                    fontsize=6.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in metrics], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT, "settings:", [k for k, _, _ in shown])


if __name__ == "__main__":
    main()
