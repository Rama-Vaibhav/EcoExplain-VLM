#!/usr/bin/env python3
"""C0.11: map of the spatial block split (sir's Major 3.3).

Recomputes the 5 x 4 block grid exactly as src/split_utils.py::write_spatial_block_splits does
(equal-width bins over the extent of the 366 patch centres) and shades each block by its role.

    .venv/bin/python scripts/review/make_spatial_split_figure.py
Output: report/paper_v3/figures/fig_spatial_split.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

OUT = ROOT.parent / "report" / "paper_v3" / "figures" / "fig_spatial_split.png"
ROLE_FILL = {"train": "#e8e8e8", "dev": "#f6d58e", "test": "#9ecae1"}
CLASS_COL = {"decline": "#c0392b", "stable": "#7f8c8d", "increase": "#27ae60"}


def main() -> None:
    man = json.loads((ROOT / "data" / "splits_spatial" / "split_manifest.json").read_text())
    parts = {s: pd.read_csv(ROOT / "data" / "splits_spatial" / f"{s}_patches.csv").assign(split=s)
             for s in ("train", "dev", "test")}
    df = pd.concat(parts.values())
    nx, ny = man["n_lon_bins"], man["n_lat_bins"]
    x0, x1, y0, y1 = df.lon.min(), df.lon.max(), df.lat.min(), df.lat.max()
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    role = {b: s for s in ("train", "dev", "test") for b in man[f"{s}_blocks"]}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for ix in range(nx):
        for iy in range(ny):
            b = ix * ny + iy
            ax.add_patch(Rectangle((xs[ix], ys[iy]), xs[ix + 1] - xs[ix], ys[iy + 1] - ys[iy],
                                   facecolor=ROLE_FILL.get(role.get(b), "white"), edgecolor="#555555", lw=0.6))
            ax.text(xs[ix] + 0.004, ys[iy + 1] - 0.004, str(b), ha="left", va="top", fontsize=7, color="#333333")
    for cls, col in CLASS_COL.items():
        sub = df[df.candidate_class == cls]
        for s, m in (("train", "o"), ("dev", "s"), ("test", "^")):
            q = sub[sub.split == s]
            ax.scatter(q.lon, q.lat, s=16 if s != "train" else 9, marker=m, c=col,
                       edgecolors="black" if s == "test" else "none", linewidths=0.4, zorder=3)
    counts = {s: len(parts[s]) for s in parts}
    handles = [Patch(facecolor=ROLE_FILL[s], edgecolor="#555555", label=f"{s} blocks ({counts[s]} patches)")
               for s in ("train", "dev", "test")]
    handles += [Line2D([], [], ls="", marker="o", color=c, label=k) for k, c in CLASS_COL.items()]
    handles += [Line2D([], [], ls="", marker=m, color="#444444", markeredgecolor="black" if s == "test" else "none",
                       label=s) for s, m in (("train", "o"), ("dev", "s"), ("test", "^"))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, frameon=False)
    ax.set_xlim(x0 - 0.01, x1 + 0.01)
    ax.set_ylim(y0 - 0.01, y1 + 0.01)
    ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
