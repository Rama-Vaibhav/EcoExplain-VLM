#!/usr/bin/env python3
"""Two-panel study-area figure for the paper (figures/fig_study_area.png).

(a) NASADEM shaded relief pulled from Google Earth Engine, with the
    processing rectangle (roi_bbox) and the 366 selected patches by class.
(b) Patch locations inside the rectangle: the 1,200-candidate pool, the
    selected patches by class and split, and the 19 human-labelled patches.

The public WDPA copy in Earth Engine has no Kanha Tiger Reserve polygon, so
the reserve boundary is not drawn.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ee  # noqa: E402

from src.paths import SELECTED_PATCHES_CSV  # noqa: E402

PAD = 0.07  # degrees of context around the rectangle
STEP = 0.0004  # output pixel size in degrees (~43 m)
CLASSES = ["decline", "stable", "increase"]
COL = {"decline": "#c0392b", "stable": "#7f8c8d", "increase": "#27ae60"}
SPLIT_MARKER = {"train": "o", "dev": "s", "test": "^"}
RECT_COLOR = "#d4ac0d"


def fetch_dem(bounds: list[float]) -> np.ndarray:
    west, south, east, north = bounds
    width = int(round((east - west) / STEP))
    height = int(round((north - south) / STEP))
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    raw = ee.data.computePixels({
        "expression": dem,
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": STEP, "shearX": 0, "translateX": west,
                "shearY": 0, "scaleY": -STEP, "translateY": north,
            },
            "crsCode": "EPSG:4326",
        },
    })
    arr = np.load(io.BytesIO(raw))
    return arr["elevation"].astype(float) if arr.dtype.names else arr.astype(float)


def load_splits() -> dict[str, str]:
    split_of = {}
    for s in SPLIT_MARKER:
        ids = pd.read_csv(ROOT / f"data/splits/{s}_patches.csv")["patch_id"]
        split_of.update({pid: s for pid in ids})
    return split_of


def load_reference_ids() -> set[str]:
    with (ROOT / "dataset.jsonl").open() as fh:
        rows = [json.loads(line) for line in fh]
    return {r["id"] for r in rows if r.get("reference_class")}


def draw_terrain(ax, fig, dem, bounds, roi, patches) -> None:
    w, s, e, n = roi
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    vmin, vmax = np.nanpercentile(dem, 1), np.nanpercentile(dem, 99) * 1.15
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(dem, cmap=plt.get_cmap("terrain"), blend_mode="soft",
                      vert_exag=0.05, vmin=vmin, vmax=vmax)
    ax.imshow(shaded, extent=extent, origin="upper", interpolation="bilinear")
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Elevation (m)")

    ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=RECT_COLOR, lw=2.2))
    for cls in CLASSES:
        sub = patches[patches["candidate_class"] == cls]
        ax.scatter(sub["lon"], sub["lat"], s=10, c=COL[cls],
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title("(a) Terrain (NASADEM) and study rectangle", fontsize=11)


def draw_patch_map(ax, roi, patches, pool, split_of, ref_ids) -> None:
    w, s, e, n = roi
    ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=RECT_COLOR, lw=2.2))
    ax.scatter(pool["lon"], pool["lat"], s=5, c="#d5d8dc", zorder=1)
    for cls in CLASSES:
        for sp, marker in SPLIT_MARKER.items():
            sub = patches[(patches["candidate_class"] == cls)
                          & (patches["patch_id"].map(split_of) == sp)]
            ax.scatter(sub["lon"], sub["lat"], s=20 if sp != "train" else 12, marker=marker,
                       c=COL[cls], edgecolor="black" if sp == "test" else "none",
                       lw=0.5, zorder=3)
    rs = patches[patches["patch_id"].isin(ref_ids)]
    ax.scatter(rs["lon"], rs["lat"], s=80, facecolor="none", edgecolor="#2c3e50",
               lw=1.2, zorder=4)
    ax.set_xlim(w - 0.01, e + 0.01)
    ax.set_ylim(s - 0.01, n + 0.01)
    ax.set_title("(b) Candidate pool and selected patches", fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pilot_kanha.yaml"))
    ap.add_argument(
        "--out",
        default=str(ROOT.parent / "report/paper_v3/figures/fig_study_area.png"),
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ee.Initialize(project=cfg["project"]["gcp_project_id"])

    roi = cfg["study_area"]["roi_bbox"]
    w, s, e, n = roi
    bounds = [w - PAD, s - PAD, e + PAD, n + PAD]
    dem = fetch_dem(bounds)
    patches = pd.read_csv(SELECTED_PATCHES_CSV)
    pool = pd.read_csv(ROOT / "data/raw_metadata/composite_stats.csv")
    split_of = load_splits()
    ref_ids = load_reference_ids()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.0, 1.08]}
    )
    draw_terrain(ax_a, fig, dem, bounds, roi, patches)
    draw_patch_map(ax_b, roi, patches, pool, split_of, ref_ids)
    aspect = 1 / np.cos(np.deg2rad((s + n) / 2))
    for ax in (ax_a, ax_b):
        ax.set_aspect(aspect)
        ax.set_xlabel("Longitude (°E)")
    ax_a.set_ylabel("Latitude (°N)")

    counts = patches["candidate_class"].value_counts()
    handles = (
        [Line2D([], [], color=RECT_COLOR, lw=2.2, label="Study rectangle"),
         Line2D([], [], ls="", marker="o", color="#d5d8dc", markersize=5,
                label=f"Candidate pool ({len(pool):,})")]
        + [Line2D([], [], ls="", marker="o", color=COL[c], markersize=6,
                  label=f"{c} ({counts[c]})") for c in CLASSES]
        + [Line2D([], [], ls="", marker=m, color="#555555",
                  markeredgecolor="black" if sp == "test" else "none",
                  markersize=6, label=sp) for sp, m in SPLIT_MARKER.items()]
        + [Line2D([], [], ls="", marker="o", markerfacecolor="none",
                  markeredgecolor="#2c3e50", markersize=9,
                  label=f"Human reference label ({len(ref_ids)})")]
    )
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04), handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
