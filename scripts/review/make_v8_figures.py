#!/usr/bin/env python3
"""Figures for report v8, drawn at print size (6.3 in wide, 7-9 pt text).

    .venv/bin/python scripts/review/make_v8_figures.py [--only study_area,workflow,...]

Writes report/paper_v3/figures/fig8_*.png:
    fig8_study_area      (a) India with Madhya Pradesh and the study area, (b) terrain and patches
    fig8_sampling        (a) candidate pool and random split, (b) spatial block split
    fig8_workflow        workflow
    fig8_ndvi_labels     (a) candidate-pool dNDVI with batch-1 percentiles, (b) the 366 patches by class
    fig8_finetuning      (a) training-class mix, (b-d) test confusion of three adapters
    fig8_explanations    class accuracy and directional claim consistency of the explanation runs

Boundaries: Natural Earth 5.1.1, admin-0 countries from India's point of view (India's official
outline) and admin-1 states, downloaded to data/reference/naturalearth/ (public domain).
Terrain: NASADEM from Earth Engine, cached in data/reference/nasadem_kanha.npz.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapefile  # noqa: E402  (pyshp)
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, REVIEW, ROOT  # noqa: E402

FIG = ROOT.parent / "report" / "paper_v3" / "figures"
REF = ROOT / "data" / "reference"
NE = REF / "naturalearth"
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/"
ROI = (80.433, 22.117, 81.050, 22.450)  # west, south, east, north (configs/pilot_kanha.yaml)
KANHA = (80.74, 22.28)

WIDTH = 6.3
COL = {"decline": "#d95f02", "stable": "#8c8c8c", "increase": "#1b7837"}
SPLIT_MARKER = {"train": "o", "dev": "s", "test": "^"}
RECT = "#c99a06"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "font.family": "sans-serif",
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=8.5)


def lonlat_aspect(ax, lat: float) -> None:
    ax.set_aspect(1 / np.cos(np.radians(lat)))


def deg_ticks(ax) -> None:
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}°N"))


def north_arrow(ax, x: float, y: float, size: float = 0.08) -> None:
    ax.annotate("N", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="top", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, mutation_scale=9))


# ---------------------------------------------------------------- data helpers
def records() -> dict[str, dict]:
    return {json.loads(line)["id"]: json.loads(line) for line in (ROOT / "dataset.jsonl").open()}


def split_ids(scheme: str = "random") -> dict[str, str]:
    d = ROOT / ("data/splits" if scheme == "random" else "data/splits_spatial")
    return {pid: s for s in SPLIT_MARKER for pid in pd.read_csv(d / f"{s}_patches.csv")["patch_id"]}


def patches_frame() -> pd.DataFrame:
    rows = records()
    pool = pd.read_csv(ROOT / "data/raw_metadata/composite_stats.csv").set_index("patch_id")
    df = pd.DataFrame({"patch_id": list(rows)})
    df["cls"] = df.patch_id.map(lambda i: rows[i]["candidate_class"])
    df["lon"] = df.patch_id.map(pool["lon"])
    df["lat"] = df.patch_id.map(pool["lat"])
    df["dndvi"] = df.patch_id.map(pool["ndvi_change"])
    df["ref"] = df.patch_id.map(lambda i: bool(rows[i].get("reference_class")))
    assert df[["lon", "lat", "dndvi"]].notna().all().all() and len(df) == 366
    return df


def ne_reader(kind: str) -> shapefile.Reader:
    name = {"admin0": "ne_10m_admin_0_countries_ind", "admin1": "ne_10m_admin_1_states_provinces"}[kind]
    folder = NE / ("admin0_ind" if kind == "admin0" else "admin1")
    if not (folder / f"{name}.shp").exists():
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(NE_URL + name + ".zip", timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(folder)
    return shapefile.Reader(str(folder / name))


def shape_path(shp) -> MPath:
    verts, codes = [], []
    parts = list(shp.parts) + [len(shp.points)]
    for a, b in zip(parts[:-1], parts[1:]):
        ring = shp.points[a:b]
        verts += ring
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 2) + [MPath.CLOSEPOLY]
    return MPath(verts, codes)


def dem_grid(bounds: tuple[float, float, float, float], step: float = 0.0004) -> np.ndarray:
    cache = REF / "nasadem_kanha.npz"
    if cache.exists():
        z = np.load(cache)
        if tuple(np.round(z["bounds"], 6)) == tuple(np.round(bounds, 6)):
            return z["dem"]
    import ee
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/pilot_kanha.yaml").read_text())
    ee.Initialize(project=cfg["project"]["gcp_project_id"])
    w, s, e, n = bounds
    raw = ee.data.computePixels({
        "expression": ee.Image("NASA/NASADEM_HGT/001").select("elevation"), "fileFormat": "NPY",
        "grid": {"dimensions": {"width": int(round((e - w) / step)), "height": int(round((n - s) / step))},
                 "affineTransform": {"scaleX": step, "shearX": 0, "translateX": w,
                                     "shearY": 0, "scaleY": -step, "translateY": n},
                 "crsCode": "EPSG:4326"}})
    arr = np.load(io.BytesIO(raw))
    dem = (arr["elevation"] if arr.dtype.names else arr).astype(float)
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, dem=dem, bounds=np.array(bounds))
    return dem


# ---------------------------------------------------------------- figure 1
def fig_study_area() -> None:
    df = patches_frame()
    fig = plt.figure(figsize=(WIDTH, 3.05))
    ax_a = fig.add_axes([0.0, 0.08, 0.36, 0.86])
    ax_b = fig.add_axes([0.44, 0.08, 0.47, 0.86])
    cax = fig.add_axes([0.925, 0.20, 0.015, 0.62])

    ext = (66.0, 99.5, 5.0, 38.5)
    r0 = ne_reader("admin0")
    f0 = [f[0] for f in r0.fields[1:]]
    ax_a.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fc="#e8f1f8", ec="none", zorder=0))
    india = None
    for i, rec in enumerate(r0.iterRecords()):
        d = dict(zip(f0, rec))
        shp = r0.shape(i)
        x0, y0, x1, y1 = shp.bbox
        if x1 < ext[0] or x0 > ext[1] or y1 < ext[2] or y0 > ext[3]:
            continue
        if d["ADM0_A3"] == "IND":
            india = shp
            continue
        ax_a.add_patch(PathPatch(shape_path(shp), fc="#ededed", ec="#b0b0b0", lw=0.3, zorder=1))
    ax_a.add_patch(PathPatch(shape_path(india), fc="#fbf3e0", ec="none", zorder=2))
    r1 = ne_reader("admin1")
    f1 = [f[0] for f in r1.fields[1:]]
    for i, rec in enumerate(r1.iterRecords()):
        d = dict(zip(f1, rec))
        if d["adm0_a3"] != "IND":
            continue
        mp = d["name"] == "Madhya Pradesh"
        ax_a.add_patch(PathPatch(shape_path(r1.shape(i)), fc="#f2c46b" if mp else "none",
                                 ec="#9a9a9a", lw=0.25, zorder=3))
    ax_a.add_patch(PathPatch(shape_path(india), fc="none", ec="#222222", lw=0.6, zorder=4))
    ax_a.plot(*KANHA, marker="s", ms=5.5, mfc="#c0392b", mec="black", mew=0.5, zorder=6)
    ax_a.annotate("Kanha Tiger\nReserve", xy=KANHA, xytext=(86.0, 14.0), fontsize=7, ha="left",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="black"), zorder=6)
    ax_a.text(77.2, 23.6, "Madhya\nPradesh", fontsize=6.5, ha="center", va="center", zorder=6)
    ax_a.text(79.2, 33.2, "INDIA", fontsize=7.5, fontweight="bold", ha="center", zorder=6, color="#444444")
    ax_a.set_xlim(ext[0], ext[1])
    ax_a.set_ylim(ext[2], ext[3])
    ax_a.set_xticks([70, 80, 90])
    ax_a.set_yticks([10, 20, 30])
    deg_ticks(ax_a)
    lonlat_aspect(ax_a, 22)
    north_arrow(ax_a, 0.93, 0.97)
    panel(ax_a, "a", "Location in India")

    w, s, e, n = ROI
    pad = 0.07
    bounds = (w - pad, s - pad, e + pad, n + pad)
    dem = dem_grid(bounds)
    vmin, vmax = np.nanpercentile(dem, 1), np.nanpercentile(dem, 99) * 1.12
    shaded = LightSource(azdeg=315, altdeg=45).shade(dem, cmap=plt.get_cmap("terrain"), blend_mode="soft",
                                                     vert_exag=0.05, vmin=vmin, vmax=vmax)
    ax_b.imshow(shaded, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], origin="upper", interpolation="bilinear")
    ax_b.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=RECT, lw=1.6, zorder=3))
    for c in CLASSES:
        q = df[df.cls == c]
        ax_b.scatter(q.lon, q.lat, s=6, c=COL[c], edgecolors="white", linewidths=0.3, zorder=4)
    ax_b.set_xlim(bounds[0], bounds[2])
    ax_b.set_ylim(bounds[1], bounds[3])
    ax_b.set_xticks([80.4, 80.6, 80.8, 81.0])
    ax_b.set_yticks([22.1, 22.2, 22.3, 22.4, 22.5])
    deg_ticks(ax_b)
    lonlat_aspect(ax_b, 22.3)
    km10 = 10 / (111.32 * np.cos(np.radians(22.3)))
    x0, y0 = bounds[0] + 0.03, bounds[1] + 0.025
    ax_b.plot([x0, x0 + km10], [y0, y0], color="black", lw=2, solid_capstyle="butt", zorder=5)
    ax_b.text(x0 + km10 / 2, y0 + 0.012, "10 km", ha="center", va="bottom", fontsize=7, zorder=5)
    north_arrow(ax_b, 0.95, 0.97)
    panel(ax_b, "b", "Terrain (NASADEM), study rectangle and 366 patches")
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Elevation (m)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    handles = ([Line2D([], [], color=RECT, lw=1.6, label="Study rectangle")]
               + [Line2D([], [], ls="", marker="o", color=COL[c], ms=4.5, label=f"{c} ({(df.cls == c).sum()})")
                  for c in CLASSES])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.66, -0.07),
               handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig8_study_area")


# ---------------------------------------------------------------- figure 2
def fig_sampling() -> None:
    df = patches_frame()
    pool = pd.read_csv(ROOT / "data/raw_metadata/composite_stats.csv")
    rnd = split_ids("random")
    sp = split_ids("spatial")
    man = json.loads((ROOT / "data/splits_spatial/split_manifest.json").read_text())
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 2.55))
    w, s, e, n = ROI

    def markers(ax, split_of: dict[str, str]) -> None:
        for c in CLASSES:
            for sname, m in SPLIT_MARKER.items():
                q = df[(df.cls == c) & (df.patch_id.map(split_of) == sname)]
                ax.scatter(q.lon, q.lat, s=7 if sname == "train" else 13, marker=m, c=COL[c],
                           edgecolors="black" if sname == "test" else "none", linewidths=0.35, zorder=3)

    ax_a.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=RECT, lw=1.3))
    ax_a.scatter(pool.lon, pool.lat, s=2.5, c="#cfd3d6", zorder=1)
    markers(ax_a, rnd)
    r = df[df.ref]
    ax_a.scatter(r.lon, r.lat, s=34, facecolor="none", edgecolor="#1f2d3a", lw=0.7, zorder=4)
    panel(ax_a, "a", "Candidate pool and random split")

    x0, x1, y0, y1 = df.lon.min(), df.lon.max(), df.lat.min(), df.lat.max()
    nx, ny = man["n_lon_bins"], man["n_lat_bins"]
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    role = {b: k for k in ("train", "dev", "test") for b in man[f"{k}_blocks"]}
    fill = {"train": "#eeeeee", "dev": "#f6d58e", "test": "#a6cee3"}
    for ix in range(nx):
        for iy in range(ny):
            b = ix * ny + iy
            ax_b.add_patch(Rectangle((xs[ix], ys[iy]), xs[ix + 1] - xs[ix], ys[iy + 1] - ys[iy],
                                     fc=fill[role[b]], ec="#666666", lw=0.5, zorder=0))
    markers(ax_b, sp)
    panel(ax_b, "b", "Spatial block split (5 × 4 blocks)")
    for ax in (ax_a, ax_b):
        ax.set_xlim(w - 0.012, e + 0.012)
        ax.set_ylim(s - 0.012, n + 0.012)
        ax.set_xticks([80.5, 80.7, 80.9])
        ax.set_yticks([22.15, 22.25, 22.35, 22.45])
        deg_ticks(ax)
        lonlat_aspect(ax, 22.3)
    counts = Counter(sp.values())
    handles = ([Line2D([], [], ls="", marker="o", color=COL[c], ms=4.5, label=c) for c in CLASSES]
               + [Line2D([], [], ls="", marker=m, color="#555555", mec="black" if k == "test" else "none",
                         ms=4.5, label=k) for k, m in SPLIT_MARKER.items()]
               + [Line2D([], [], ls="", marker="o", color="#cfd3d6", ms=3.5, label="candidate pool (1,200)"),
                  Line2D([], [], ls="", marker="o", mfc="none", mec="#1f2d3a", ms=6, label="human check (19)")]
               + [Patch(fc=fill[k], ec="#666666", label=f"{k} blocks ({counts[k]})") for k in ("train", "dev", "test")])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.17),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_sampling")


# ---------------------------------------------------------------- figure 3
def fig_workflow() -> None:
    fig = plt.figure(figsize=(WIDTH, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    pal = {"in": ("#e3f1dc", "#4f8a3c"), "proc": ("#dde8f6", "#3d6a99"), "bench": ("#f8e7cc", "#b7792e"),
           "split": ("#ebe1f5", "#6f52a1"), "eval": ("#fbe3e1", "#a8463d"), "stat": ("#eeeeee", "#555555")}

    def box(x, y, w, h, kind, title, body=""):
        fc, ec = pal[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        if body:
            ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color="#1b1f27")
            ax.text(x + w / 2, y + h / 2 - 2.6, body, ha="center", va="center", fontsize=6.9, color="#333a44",
                    linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.4, color="#1b1f27",
                    linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#6b7380"))

    def stage(y0, y1, text):
        ax.text(1.6, (y0 + y1) / 2, text, ha="center", va="center", rotation=90, fontsize=7.2,
                fontweight="bold", color="#1b1f27")

    L, R = 5.0, 99.0
    span = R - L

    def cols(n, gap):
        w = (span - gap * (n - 1)) / n
        return [L + i * (w + gap) for i in range(n)], w

    xs, w5 = cols(5, 1.6)
    stage(80, 95, "Data")
    for x, (t, b) in zip(xs, [("Sentinel-2 L2A", "RGB chips; NDVI,\nEVI, NDMI, NBR"),
                               ("Sentinel-1 GRD", "VV and VH\nbackscatter"),
                               ("CHIRPS", "window\nrainfall"),
                               ("ERA5-Land", "temperature and soil-\nmoisture anomalies"),
                               ("Static layers", "NASADEM terrain;\nWorldCover mask")]):
        box(x, 80, w5, 15, "in", t, b)
        arrow(x + w5 / 2, 79.6, x + w5 / 2, 74.4)
    stage(62.5, 73.5, "Earth Engine")
    box(L, 62.5, span, 11.2, "proc",
        "Cloud-masked median composites for T0 (Jan–Feb 2022) and T1 (Jan–Feb 2023)\n"
        "640 m grid: 5,800 cells → 3,668 forest cells → 1,200 scored candidates")
    xs3, w3 = cols(3, 2.5)
    for x in xs3:
        arrow(x + w3 / 2, 62.1, x + w3 / 2, 57.4)
    stage(43.5, 57, "Benchmark")
    for x, (t, b) in zip(xs3, [("Paired RGB chips", "T0 and T1, 640 m patches"),
                                ("Relative ΔNDVI label", "15th/85th percentiles,\ntwo selection batches"),
                                ("Evidence card", "full (leaky) | evidence-only |\ncompact table")]):
        box(x, 43.5, w3, 13.5, "bench", t, b)
        arrow(x + w3 / 2, 43.1, x + w3 / 2, 39.0)
    stage(30.3, 38.5, "Splits")
    box(L, 30.3, span, 8.2, "split", "366 patches · random split 256 / 55 / 55 · spatial block split 266 / 46 / 54")
    for x in xs3:
        arrow(x + w3 / 2, 29.9, x + w3 / 2, 25.6)
    stage(11.6, 25.2, "Evaluation")
    for x, (t, b) in zip(xs3, [("A  Zero-shot classification", "4 VLMs; full card vs evidence-\nonly vs compact; both splits"),
                                ("B  Class LoRA", "16 balanced examples; 7 adapters;\nmatched zero-shot control"),
                                ("C  Explanations", "claim checker; explanation LoRA;\nmatched zero-shot control")]):
        box(x, 11.6, w3, 13.6, "eval", t, b)
        arrow(x + w3 / 2, 11.2, x + w3 / 2, 8.4)
    stage(0.8, 7.8, "Tests")
    box(L, 0.8, span, 7, "stat", "Accuracy with Wilson 95% CI, macro-F1, exact McNemar tests with Holm adjustment")
    save(fig, "fig8_workflow")


# ---------------------------------------------------------------- figure 4
def fig_ndvi_labels() -> None:
    pool = pd.read_csv(ROOT / "data/raw_metadata/composite_stats.csv")
    df = patches_frame()
    q15, q85 = pool.ndvi_change.quantile(0.15), pool.ndvi_change.quantile(0.85)
    assert round(q15, 4) == -0.0377 and round(q85, 4) == 0.0031
    bins = np.arange(-0.09, 0.0801, 0.004)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 2.45))
    ax_a.hist(pool.ndvi_change, bins=bins, color="#b8bec4")
    ax_a.axvline(q15, color=COL["decline"], ls="--", lw=1.1)
    ax_a.axvline(q85, color=COL["increase"], ls="--", lw=1.1)
    ax_a.axvline(0, color="black", lw=0.6)
    ax_a.text(q15 - 0.003, 168, f"15th\npercentile\n{q15:.4f}".replace("-", "−"), ha="right", va="top",
              fontsize=6.8, color=COL["decline"])
    ax_a.text(q85 + 0.003, 168, f"85th\npercentile\n+{q85:.4f}", ha="left", va="top", fontsize=6.8,
              color=COL["increase"])
    ax_a.set_ylabel("Number of patches")
    ax_a.set_ylim(0, 175)
    panel(ax_a, "a", f"Candidate pool ({len(pool):,})")
    for c in CLASSES:
        ax_b.hist(df[df.cls == c].dndvi, bins=bins, color=COL[c], alpha=0.75, label=f"{c} ({(df.cls == c).sum()})")
    ax_b.axvline(0, color="black", lw=0.6)
    ax_b.legend(frameon=False, loc="upper right", fontsize=6.8)
    ax_b.set_ylim(0, 46)
    panel(ax_b, "b", "Selected patches by class (366)")
    for ax in (ax_a, ax_b):
        ax.set_xlabel("Patch-mean ΔNDVI (2023 − 2022)")
        ax.set_xlim(-0.09, 0.08)
        ax.set_xticks([-0.08, -0.04, 0.0, 0.04, 0.08])
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_ndvi_labels")


# ---------------------------------------------------------------- figure 5
def confusion(path: Path) -> np.ndarray:
    m = np.zeros((3, 3), int)
    for x in json.loads(path.read_text())["predictions"]:
        if x.get("pred") in CLASSES:
            m[CLASSES.index(x["gold"]), CLASSES.index(x["pred"])] += 1
    return m


def fig_finetuning() -> None:
    rows = records()
    train = list(pd.read_csv(ROOT / "data/splits/train_patches.csv")["patch_id"])
    first16 = Counter(rows[i]["candidate_class"] for i in train[:16])
    targets = [json.loads(line) for line in (ROOT / "data/explain_targets/train_targets.jsonl").open()]
    first128 = set(train[:128])
    expl = Counter(t["change_class"] for t in targets if t["id"] in first128)
    bal = Counter({"decline": 6, "stable": 5, "increase": 5})  # Ada job 29575 log: train class mix
    assert [first16[c] for c in CLASSES] == [9, 6, 1] and [expl[c] for c in CLASSES] == [16, 0, 0]
    res = ROOT / "results"
    cms = [(confusion(res / "vlm_combo_qwen_vl_finetuned_ecoexplain.json"), "b", "Class LoRA, full card\n(superseded): 67.3%"),
           (confusion(res / "protocol/qwen_balanced_leakfree_lora.json"), "c", "Class LoRA, balanced,\nevidence-only: 36.4%"),
           (confusion(res / "explain/explain_qwen_vl_finetuned_ecoexplain.json"), "d", "Explanation LoRA,\ncompact table: 61.8%")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0))
    ax = axes[0, 0]
    x = np.arange(3)
    for off, mix, hatch, lab in [(-0.27, first16, "", "Class LoRA, first 16 by ID"),
                                 (0.0, bal, "....", "Class LoRA, balanced"),
                                 (0.27, expl, "////", "Explanation LoRA")]:
        ax.bar(x + off, [mix[c] for c in CLASSES], 0.25, color=[COL[c] for c in CLASSES], edgecolor="black",
               lw=0.5, hatch=hatch, label=lab)
        for xi, c in zip(x, CLASSES):
            ax.text(xi + off, mix[c] + 0.3, str(mix[c]), ha="center", fontsize=7)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 21)
    ax.set_ylabel("Training examples")
    leg = ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    for h in leg.legend_handles:
        h.set_facecolor("white")
    panel(ax, "a", "Classes of the 16 training examples")
    for a, (m, letter, title) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], cms):
        a.imshow(m, cmap="Oranges", vmin=0, vmax=19)
        for i in range(3):
            for j in range(3):
                a.text(j, i, m[i, j], ha="center", va="center", fontsize=8,
                       color="white" if m[i, j] > 12 else "black")
        a.set_xticks(range(3), CLASSES)
        a.set_yticks(range(3), CLASSES)
        a.set_xlabel("Predicted class")
        a.set_ylabel("Label")
        a.spines[["top", "right"]].set_visible(True)
        panel(a, letter, title)
    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    save(fig, "fig8_finetuning")


# ---------------------------------------------------------------- figure 6
def fig_explanations() -> None:
    stats = json.loads((REVIEW / "stats_report.json").read_text())["runs"]
    claims = json.loads((REVIEW / "claim_audit.json").read_text())
    settings = [("llava_freeform", "LLaVA, free-form", "#b4b9bf"),
                ("qwen_expl_zs_384", "Qwen zero-shot, free-form (384 px)", "#6c757d"),
                ("expl_lora", "Explanation LoRA (384 px)", "#c8691c"),
                ("qwen_struct", "Qwen, structured", "#2171b5"),
                ("llava_struct", "LLaVA, structured", "#9ecae1")]
    metrics = [("Class accuracy", lambda r, c: (r["k"], r["n"])),
               ("Claim consistency,\nall scored claims", lambda r, c: (c["consistent"], c["scored"])),
               ("Claim consistency,\nwithout copied rule text",
                lambda r, c: (c["consistent_excluding_rule"], c["scored_excluding_rule"]))]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    width = 0.16
    x = np.arange(len(metrics))
    for i, (k, lab, col) in enumerate(settings):
        vals = [f(stats[k], claims[k]) for _, f in metrics]
        bars = ax.bar(x + (i - 2) * width, [a / b for a, b in vals], width, color=col, edgecolor="#333333",
                      lw=0.4, label=lab)
        for b, (a, nn) in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{a}/{nn}", ha="center", va="bottom",
                    fontsize=6.2, rotation=90)
    ax.set_xticks(x, [m for m, _ in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    save(fig, "fig8_explanations")


FIGS = {"study_area": fig_study_area, "sampling": fig_sampling, "workflow": fig_workflow,
        "ndvi_labels": fig_ndvi_labels, "finetuning": fig_finetuning, "explanations": fig_explanations}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(FIGS))
    for name in ap.parse_args().only.split(","):
        FIGS[name]()
