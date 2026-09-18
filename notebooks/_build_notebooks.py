"""Regenerate EcoExplain-VLM notebooks. Run: python notebooks/_build_notebooks.py"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text if text.endswith("\n") else text + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text if text.endswith("\n") else text + "\n",
    }


def write_nb(name: str, cells: list) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", path)


nb01 = [
    md(
        """# 01 — Five-source GEE extraction (Kanha Tiger Reserve)

Builds the **complete EcoExplain-VLM cube** per patch, matching `Dataset_Extraction.ipynb`:

| # | Source | Variables exported |
|---|---|---|
| 1 | **Sentinel-2 SR** | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + NDVI, EVI, NDMI, NBR + RGB |
| 2 | **Sentinel-1 GRD** | VV, VH (IW, dominant orbit pass) |
| 3 | **CHIRPS** | Window rainfall total (mm) |
| 4 | **ERA5-Land** | Month-matched **temperature** and **soil-moisture** anomalies (climatology 2015–2021) |
| 5 | **NASADEM** | Elevation, slope, aspect |

**Dates (same as the parent notebook):**

- **T0:** 2022-01-01 → 2022-02-28
- **T1:** 2023-01-01 → 2023-02-28

**ROI:** Kanha official bbox `[80.433, 22.117, 81.050, 22.450]`

**Patches:** `data/raw_metadata/selected_patches.csv` (36 class-balanced chips).

Each patch is written to `data/raw_tiffs/` as:

```
{id}_T0_rgb.tif / {id}_T1_rgb.tif
{id}_T0_s2_optical.tif / {id}_T1_s2_optical.tif     # 10 S2 bands
{id}_T0_indices.tif / {id}_T1_indices.tif           # NDVI, EVI, NDMI, NBR
{id}_T0_s1.tif / {id}_T1_s1.tif                     # VV, VH
{id}_T0_rainfall.tif / {id}_T1_rainfall.tif
{id}_T0_era5.tif / {id}_T1_era5.tif                 # temp_anomaly, soil_moisture_anomaly
{id}_dem.tif                                        # elevation, slope, aspect
{id}_delta_ndvi.tif / {id}_delta_nbr.tif
```
"""
    ),
    md("## 1. Authenticate Earth Engine"),
    code(
        r"""from pathlib import Path
import json
import sys

import ee
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.gee_utils import (
    CHIRPS_SCALE,
    DEM_SCALE,
    ERA5_SCALE,
    S2_INDICES,
    S2_OPTICAL_BANDS,
    get_chirps_rainfall,
    get_dem_terrain,
    get_dominant_orbit_pass,
    get_era5_anomalies,
    get_s1_composite,
    get_s2_composite,
    masked_fraction,
    patch_geometry,
    reduce_coarse,
    reduce_fine,
    rgb_reflectance,
)

RAW_TIFFS = ROOT / "data" / "raw_tiffs"
RAW_META = ROOT / "data" / "raw_metadata"
RAW_TIFFS.mkdir(parents=True, exist_ok=True)

GCP_PROJECT_ID = "chintaphaniramavaibhav1"

try:
    ee.Initialize(project=GCP_PROJECT_ID)
    print("Earth Engine already initialized.")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GCP_PROJECT_ID)
    print("Earth Engine authenticated and initialized.")
print("ROOT:", ROOT)
"""
    ),
    md("## 2. Kanha ROI, matched seasonal windows, patch size"),
    code(
        r"""roi_bbox = [80.433, 22.117, 81.050, 22.450]
roi = ee.Geometry.Rectangle(roi_bbox)

BEFORE_START, BEFORE_END = "2022-01-01", "2022-02-28"
AFTER_START, AFTER_END = "2023-01-01", "2023-02-28"
CLIM_START, CLIM_END = "2015-01-01", "2021-12-31"

PATCH_PIXELS = 64
PIXEL_SIZE_M = 10
PATCH_SIZE_M = PATCH_PIXELS * PIXEL_SIZE_M
MAX_SCENE_CLOUD_PCT = 20
MAX_PATCHES = 36
USE_DRIVE_EXPORT = False
DRIVE_FOLDER = "ecoexplain_vlm_raw_tiffs"

print("ROI area (sq km):", roi.area().divide(1e6).getInfo())
print(f"T0: {BEFORE_START} -> {BEFORE_END}")
print(f"T1: {AFTER_START} -> {AFTER_END}")
print(f"ERA5 climatology: {CLIM_START} -> {CLIM_END}")
print(f"S2 optical bands: {S2_OPTICAL_BANDS}")
"""
    ),
    md("## 3. Build all five source composites"),
    code(
        r"""s2_t0, n_s2_t0 = get_s2_composite(roi, BEFORE_START, BEFORE_END, MAX_SCENE_CLOUD_PCT)
s2_t1, n_s2_t1 = get_s2_composite(roi, AFTER_START, AFTER_END, MAX_SCENE_CLOUD_PCT)

S1_ORBIT_PASS = get_dominant_orbit_pass(roi, BEFORE_START, AFTER_END)
s1_t0, n_s1_t0 = get_s1_composite(roi, BEFORE_START, BEFORE_END, S1_ORBIT_PASS)
s1_t1, n_s1_t1 = get_s1_composite(roi, AFTER_START, AFTER_END, S1_ORBIT_PASS)

chirps_t0, n_ch_t0 = get_chirps_rainfall(roi, BEFORE_START, BEFORE_END)
chirps_t1, n_ch_t1 = get_chirps_rainfall(roi, AFTER_START, AFTER_END)

era5_temp_t0, era5_sm_t0 = get_era5_anomalies(roi, BEFORE_START, BEFORE_END, CLIM_START, CLIM_END)
era5_temp_t1, era5_sm_t1 = get_era5_anomalies(roi, AFTER_START, AFTER_END, CLIM_START, CLIM_END)
era5_t0 = era5_temp_t0.addBands(era5_sm_t0)
era5_t1 = era5_temp_t1.addBands(era5_sm_t1)

dem = get_dem_terrain(roi)

t0_rgb = rgb_reflectance(s2_t0)
t1_rgb = rgb_reflectance(s2_t1)
delta_ndvi = s2_t1.select("NDVI").subtract(s2_t0.select("NDVI")).rename("delta_ndvi")
delta_nbr = s2_t1.select("NBR").subtract(s2_t0.select("NBR")).rename("delta_nbr")

print("S2 scenes T0/T1:", n_s2_t0.getInfo(), n_s2_t1.getInfo())
print("S1 orbit:", S1_ORBIT_PASS, "scenes T0/T1:", n_s1_t0.getInfo(), n_s1_t1.getInfo())
print("CHIRPS days T0/T1:", n_ch_t0.getInfo(), n_ch_t1.getInfo())
print("Five-source composites ready.")
"""
    ),
    md("## 4. Load the 36-patch pilot list"),
    code(
        r"""csv_path = RAW_META / "selected_patches.csv"
if not csv_path.exists():
    raise FileNotFoundError(csv_path)

selected = pd.read_csv(csv_path).head(MAX_PATCHES)
print(selected["candidate_class"].value_counts())
print("patches:", len(selected))
"""
    ),
    md(
        """## 5. Export GeoTIFFs (all five sources)

Local `geemap.ee_export_image` writes into `data/raw_tiffs/`. If a local export fails, that patch falls back to Drive. Set `USE_DRIVE_EXPORT = True` to queue everything as EE tasks.
"""
    ),
    code(
        r"""import geemap


def export_specs(pid, geom):
    return [
        (t0_rgb.clip(geom), RAW_TIFFS / f"{pid}_T0_rgb.tif", PIXEL_SIZE_M),
        (t1_rgb.clip(geom), RAW_TIFFS / f"{pid}_T1_rgb.tif", PIXEL_SIZE_M),
        (s2_t0.select(S2_OPTICAL_BANDS).clip(geom), RAW_TIFFS / f"{pid}_T0_s2_optical.tif", PIXEL_SIZE_M),
        (s2_t1.select(S2_OPTICAL_BANDS).clip(geom), RAW_TIFFS / f"{pid}_T1_s2_optical.tif", PIXEL_SIZE_M),
        (s2_t0.select(S2_INDICES).clip(geom), RAW_TIFFS / f"{pid}_T0_indices.tif", PIXEL_SIZE_M),
        (s2_t1.select(S2_INDICES).clip(geom), RAW_TIFFS / f"{pid}_T1_indices.tif", PIXEL_SIZE_M),
        (s1_t0.select(["VV", "VH"]).clip(geom), RAW_TIFFS / f"{pid}_T0_s1.tif", PIXEL_SIZE_M),
        (s1_t1.select(["VV", "VH"]).clip(geom), RAW_TIFFS / f"{pid}_T1_s1.tif", PIXEL_SIZE_M),
        (chirps_t0.clip(geom), RAW_TIFFS / f"{pid}_T0_rainfall.tif", CHIRPS_SCALE),
        (chirps_t1.clip(geom), RAW_TIFFS / f"{pid}_T1_rainfall.tif", CHIRPS_SCALE),
        (era5_t0.clip(geom), RAW_TIFFS / f"{pid}_T0_era5.tif", ERA5_SCALE),
        (era5_t1.clip(geom), RAW_TIFFS / f"{pid}_T1_era5.tif", ERA5_SCALE),
        (dem.clip(geom), RAW_TIFFS / f"{pid}_dem.tif", DEM_SCALE),
        (delta_ndvi.clip(geom), RAW_TIFFS / f"{pid}_delta_ndvi.tif", PIXEL_SIZE_M),
        (delta_nbr.clip(geom), RAW_TIFFS / f"{pid}_delta_nbr.tif", PIXEL_SIZE_M),
    ]


def export_local(pid, geom):
    for image, out_path, scale in export_specs(pid, geom):
        if out_path.exists():
            print("skip", out_path.name)
            continue
        geemap.ee_export_image(
            image,
            filename=str(out_path),
            scale=scale,
            region=geom,
            file_per_band=False,
        )
        print("wrote", out_path.name)


def export_drive(pid, geom):
    tasks = []
    for image, out_path, scale in export_specs(pid, geom):
        name = out_path.stem
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=name[:100],
            folder=DRIVE_FOLDER,
            fileNamePrefix=name,
            region=geom,
            scale=scale,
            maxPixels=1e8,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append(task)
    return tasks


all_tasks = []
for _, row in selected.iterrows():
    pid = str(row["patch_id"])
    geom = patch_geometry(float(row["lon"]), float(row["lat"]), PATCH_SIZE_M, roi)
    if USE_DRIVE_EXPORT:
        all_tasks.extend(export_drive(pid, geom))
        print("queued Drive", pid)
        continue
    try:
        export_local(pid, geom)
    except Exception as exc:
        print(f"local export failed {pid}: {exc} -> Drive fallback")
        all_tasks.extend(export_drive(pid, geom))

print("Drive tasks started:", len(all_tasks))
print("local tifs:", len(list(RAW_TIFFS.glob("*.tif"))))
"""
    ),
    md(
        """## 5b. Download from Google Drive (if you used Drive export)

**Option A — Automated (OAuth):**
```bash
# From project root (not notebooks/)
python scripts/02_download_drive.py
```
Place `credentials.json` in the project root first (see README).

**Option B — Manual from drive.google.com:**
1. Open https://drive.google.com → folder `ecoexplain_vlm_raw_tiffs`
2. Select all `.tif` files → Download → unzip into `data/drive_inbox/`
3. Run:
```bash
python scripts/02_download_drive.py --from-inbox
```

**Option C — Full CLI pipeline:**
```bash
python scripts/run_pipeline.py
```
"""
    ),
    md("## 6. Patch-level scalars (fixes the old ERA5/CHIRPS/DEM `None` bug)"),
    code(
        r"""def compute_patch_scalars(lon, lat):
    geom = patch_geometry(lon, lat, PATCH_SIZE_M, roi)

    def s2_mean(img, band):
        return img.select(band).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=PIXEL_SIZE_M, maxPixels=1e8
        ).get(band)

    keys = [
        "ndvi_t0", "ndvi_t1", "evi_t0", "evi_t1", "ndmi_t0", "ndmi_t1", "nbr_t0", "nbr_t1",
        "vv_t0", "vv_t1", "vh_t0", "vh_t1",
        "masked_frac_t0", "masked_frac_t1",
        "era5_temp_anomaly_t0", "era5_temp_anomaly_t1",
        "era5_sm_anomaly_t0", "era5_sm_anomaly_t1",
        "chirps_rainfall_t0", "chirps_rainfall_t1",
        "elevation_m", "slope_deg", "aspect_deg",
    ]
    values = ee.List([
        s2_mean(s2_t0, "NDVI"), s2_mean(s2_t1, "NDVI"),
        s2_mean(s2_t0, "EVI"), s2_mean(s2_t1, "EVI"),
        s2_mean(s2_t0, "NDMI"), s2_mean(s2_t1, "NDMI"),
        s2_mean(s2_t0, "NBR"), s2_mean(s2_t1, "NBR"),
        s2_mean(s1_t0, "VV"), s2_mean(s1_t1, "VV"),
        s2_mean(s1_t0, "VH"), s2_mean(s1_t1, "VH"),
        masked_fraction(s2_t0.clip(geom), geom),
        masked_fraction(s2_t1.clip(geom), geom),
        reduce_coarse(era5_temp_t0, lon, lat, "temp_anomaly", ERA5_SCALE),
        reduce_coarse(era5_temp_t1, lon, lat, "temp_anomaly", ERA5_SCALE),
        reduce_coarse(era5_sm_t0, lon, lat, "soil_moisture_anomaly", ERA5_SCALE),
        reduce_coarse(era5_sm_t1, lon, lat, "soil_moisture_anomaly", ERA5_SCALE),
        reduce_coarse(chirps_t0, lon, lat, "rainfall_mm", CHIRPS_SCALE),
        reduce_coarse(chirps_t1, lon, lat, "rainfall_mm", CHIRPS_SCALE),
        reduce_fine(dem, geom, "elevation_m", DEM_SCALE),
        reduce_fine(dem, geom, "slope_deg", DEM_SCALE),
        reduce_fine(dem, geom, "aspect_deg", DEM_SCALE),
    ]).getInfo()
    out = dict(zip(keys, values))
    out["s1_orbit_pass"] = S1_ORBIT_PASS
    out["s1_n_t0"] = n_s1_t0.getInfo()
    out["s1_n_t1"] = n_s1_t1.getInfo()
    out["t0_start"], out["t0_end"] = BEFORE_START, BEFORE_END
    out["t1_start"], out["t1_end"] = AFTER_START, AFTER_END
    return out


scalar_path = RAW_META / "patch_scalars.json"
scalars = {}
if scalar_path.exists():
    scalars = json.loads(scalar_path.read_text())

# Reuse parent CSV values when present so you are not blocked on extra EE getInfo calls.
csv_alias = {
    "ndvi_before": "ndvi_t0", "ndvi_after": "ndvi_t1",
    "evi_before": "evi_t0", "evi_after": "evi_t1",
    "ndmi_before": "ndmi_t0", "ndmi_after": "ndmi_t1",
    "nbr_before": "nbr_t0", "nbr_after": "nbr_t1",
    "vv_before": "vv_t0", "vv_after": "vv_t1",
    "vh_before": "vh_t0", "vh_after": "vh_t1",
    "masked_frac_before": "masked_frac_t0", "masked_frac_after": "masked_frac_t1",
    "era5_temp_anomaly_before": "era5_temp_anomaly_t0",
    "era5_temp_anomaly_after": "era5_temp_anomaly_t1",
    "era5_sm_anomaly_before": "era5_sm_anomaly_t0",
    "era5_sm_anomaly_after": "era5_sm_anomaly_t1",
    "chirps_rainfall_before": "chirps_rainfall_t0",
    "chirps_rainfall_after": "chirps_rainfall_t1",
    "s1_n_before": "s1_n_t0", "s1_n_after": "s1_n_t1",
}

for _, row in selected.iterrows():
    pid = str(row["patch_id"])
    rec = {new: row[old] for old, new in csv_alias.items() if old in row.index}
    for col in ("elevation_m", "slope_deg", "aspect_deg", "s1_orbit_pass", "fire_detected", "candidate_class"):
        if col in row.index:
            rec[col] = row[col]
    rec["t0_start"], rec["t0_end"] = BEFORE_START, BEFORE_END
    rec["t1_start"], rec["t1_end"] = AFTER_START, AFTER_END
    scalars[pid] = rec

scalar_path.write_text(json.dumps(scalars, indent=2, default=str))
print("wrote", scalar_path, "n=", len(scalars))
print("sample keys", sorted(next(iter(scalars.values())).keys())[:12], "...")
"""
    ),
    md("## 7. Inventory"),
    code(
        r"""from collections import Counter
stems = [p.name.split("_", 1)[-1] if "_" in p.name else p.name for p in RAW_TIFFS.glob("*.tif")]
print("tif count", len(list(RAW_TIFFS.glob("*.tif"))))
print(Counter(stems).most_common(20))
"""
    ),
]

nb02 = [
    md(
        """# 02 — DIP + five-source packaging

Otsu on **ΔNDVI** (deforestation) and **ΔNBR** (fire) → bounding boxes on the **512×512** VLM canvas.

All five sources are padded with the **same** transform so tensors align with the RGB chips:

1. Sentinel-2 10-band optical + indices
2. Sentinel-1 VV / VH
3. CHIRPS rainfall
4. ERA5-Land temperature & soil-moisture anomalies
5. DEM elevation / slope / aspect
"""
    ),
    md("## 1. Imports"),
    code(
        r"""from pathlib import Path
import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.dip_utils import (
    SCENE_SUFFIXES,
    apply_reference_class_policy,
    array_nanmean,
    build_disturbance_triplets,
    discover_scenes,
    load_geotiff_hwc,
    load_rgb_geotiff,
    pad_and_resize,
    percentile_stretch_rgb,
    process_tensor_otsu,
    read_single_band,
)

RAW_TIFFS = ROOT / "data" / "raw_tiffs"
PROC = ROOT / "data" / "processed_images"
MASKS = ROOT / "data" / "evidence_masks"
META = ROOT / "data" / "raw_metadata"
TRIPLETS_JSON = META / "triplets.json"
FINAL_LABELS = ROOT.parent / "data" / "labels" / "final_labels.csv"
PROC.mkdir(parents=True, exist_ok=True)
MASKS.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = (512, 512)
MIN_DROP = 0.04  # ignore tiny spectral noise before Otsu
print("discovered suffix keys", list(SCENE_SUFFIXES))
"""
    ),
    md("## 2. Discover complete scenes"),
    code(
        r"""scenes = discover_scenes(RAW_TIFFS)
print("complete DIP scenes:", len(scenes))
if scenes:
    print("example keys", sorted(k for k in scenes[0] if k != "scene_id"))
"""
    ),
    md("## 3. Demo cube if GEE TIFFs are not downloaded yet"),
    code(
        r"""import rasterio
from rasterio.transform import from_origin


def _write(path, arr, transform):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    profile = dict(
        driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype="float32", crs="EPSG:4326", transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def write_demo_cube(raw_dir: Path, scene_id="demo_kanha", h=48, w=72):
    rng = np.random.default_rng(7)
    transform = from_origin(80.7, 22.3, 10, 10)
    y0, y1, x0, x1 = 18, 40, 38, 64
    t0_rgb = np.clip(0.18 + 0.12 * rng.random((h, w, 3)), 0, 1)
    t1_rgb = t0_rgb.copy()
    t1_rgb[y0:y1, x0:x1, 1] *= 0.35
    optical_t0 = np.clip(0.1 + 0.2 * rng.random((h, w, 10)), 0, 1)
    optical_t1 = optical_t0.copy()
    optical_t1[y0:y1, x0:x1, 2] *= 0.5
    indices_t0 = rng.normal(0.6, 0.05, (h, w, 4))
    indices_t1 = indices_t0.copy()
    indices_t1[y0:y1, x0:x1, 0] -= 0.35
    indices_t1[y0:y1, x0:x1, 3] -= 0.42
    s1_t0 = rng.normal(-10, 1.5, (h, w, 2))
    s1_t1 = s1_t0 + rng.normal(0, 0.4, (h, w, 2))
    rain_t0 = np.full((h, w), 72.0)
    rain_t1 = np.full((h, w), 25.0)
    era5_t0 = np.dstack([np.full((h, w), -1.5), np.full((h, w), 0.02)])
    era5_t1 = np.dstack([np.full((h, w), 0.9), np.full((h, w), -0.04)])
    dem = np.dstack([np.full((h, w), 650.0), np.full((h, w), 6.0), np.full((h, w), 180.0)])
    d_ndvi = np.zeros((h, w), np.float32)
    d_nbr = np.zeros((h, w), np.float32)
    d_ndvi[y0:y1, x0:x1] = -0.35
    d_nbr[y0:y1, x0:x1] = -0.42
    d_ndvi += rng.normal(0, 0.02, (h, w)).astype(np.float32)
    d_nbr += rng.normal(0, 0.02, (h, w)).astype(np.float32)
    mapping = {
        f"{scene_id}_T0_rgb.tif": t0_rgb,
        f"{scene_id}_T1_rgb.tif": t1_rgb,
        f"{scene_id}_T0_s2_optical.tif": optical_t0,
        f"{scene_id}_T1_s2_optical.tif": optical_t1,
        f"{scene_id}_T0_indices.tif": indices_t0,
        f"{scene_id}_T1_indices.tif": indices_t1,
        f"{scene_id}_T0_s1.tif": s1_t0,
        f"{scene_id}_T1_s1.tif": s1_t1,
        f"{scene_id}_T0_rainfall.tif": rain_t0,
        f"{scene_id}_T1_rainfall.tif": rain_t1,
        f"{scene_id}_T0_era5.tif": era5_t0,
        f"{scene_id}_T1_era5.tif": era5_t1,
        f"{scene_id}_dem.tif": dem,
        f"{scene_id}_delta_ndvi.tif": d_ndvi,
        f"{scene_id}_delta_nbr.tif": d_nbr,
    }
    for name, arr in mapping.items():
        _write(raw_dir / name, arr, transform)
    print("wrote demo five-source cube", scene_id, f"{h}x{w}")


if not scenes:
    write_demo_cube(RAW_TIFFS)
    scenes = discover_scenes(RAW_TIFFS)
    print("using demo cube until GEE exports land in data/raw_tiffs/")
"""
    ),
    md("## 4. Otsu, boxes, aligned 512×512 tensors"),
    code(
        r"""csv_path = META / "selected_patches.csv"
patch_csv = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
scalars_path = META / "patch_scalars.json"
scalars = json.loads(scalars_path.read_text()) if scalars_path.exists() else {}

GEO_COLS = {"lon", "lat", "patch_id"}


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def save_npy(path, arr):
    np.save(path, arr)
    return rel(path)


def pad_stack(path, native_boxes=None):
    hwc = load_geotiff_hwc(path)
    if hwc.shape[-1] == 1:
        hwc = hwc[..., 0]
    out, boxes, meta = pad_and_resize(hwc, TARGET_SIZE, bboxes=native_boxes, keep_dtype=True)
    return out.astype(np.float32), boxes, meta


records = []
final_labels = pd.read_csv(FINAL_LABELS) if FINAL_LABELS.exists() else pd.DataFrame()
ref_map = dict(zip(final_labels.patch_id.astype(str), final_labels.reference_class)) if not final_labels.empty else {}

for scene in scenes:
    sid = scene["scene_id"]
    ndvi_mask = process_tensor_otsu(scene["delta_ndvi"], min_drop=MIN_DROP)
    nbr_mask = process_tensor_otsu(scene["delta_nbr"], min_drop=MIN_DROP)
    delta_ndvi = read_single_band(scene["delta_ndvi"])
    delta_nbr = read_single_band(scene["delta_nbr"])
    native_triplets = build_disturbance_triplets(ndvi_mask, nbr_mask, delta_ndvi, delta_nbr)
    native_triplets = apply_reference_class_policy(native_triplets, ref_map.get(sid))
    native_boxes = [t[0] for t in native_triplets]

    t0_rgb = percentile_stretch_rgb(load_rgb_geotiff(scene["t0_rgb"]))
    t1_rgb = percentile_stretch_rgb(load_rgb_geotiff(scene["t1_rgb"]))
    t0_512, boxes_512, meta = pad_and_resize(t0_rgb, TARGET_SIZE, bboxes=native_boxes)
    t1_512, _, _ = pad_and_resize(t1_rgb, TARGET_SIZE, bboxes=native_boxes)
    vlm_triplets = [[b, t[1], t[2]] for b, t in zip(boxes_512, native_triplets)]

    t0_jpg = PROC / f"{sid}_T0.jpg"
    t1_jpg = PROC / f"{sid}_T1.jpg"
    Image.fromarray(t0_512).save(t0_jpg, quality=95)
    Image.fromarray(t1_512).save(t1_jpg, quality=95)

    ndvi_512, _, _ = pad_and_resize((ndvi_mask * 255).astype(np.uint8), TARGET_SIZE)
    nbr_512, _, _ = pad_and_resize((nbr_mask * 255).astype(np.uint8), TARGET_SIZE)
    np.save(MASKS / f"{sid}_delta_ndvi_mask.npy", ndvi_512)
    np.save(MASKS / f"{sid}_delta_nbr_mask.npy", nbr_512)

    paths = {
        "t0_image": rel(t0_jpg),
        "t1_image": rel(t1_jpg),
        "delta_ndvi_mask": rel(MASKS / f"{sid}_delta_ndvi_mask.npy"),
        "delta_nbr_mask": rel(MASKS / f"{sid}_delta_nbr_mask.npy"),
    }
    tensor_map = {
        "t0_optical": "t0_optical",
        "t1_optical": "t1_optical",
        "t0_indices": "t0_indices",
        "t1_indices": "t1_indices",
        "t0_s1": "t0_s1",
        "t1_s1": "t1_s1",
        "t0_rainfall": "t0_rainfall",
        "t1_rainfall": "t1_rainfall",
        "t0_era5": "t0_era5",
        "t1_era5": "t1_era5",
        "dem": "dem",
        "delta_ndvi": "delta_ndvi",
        "delta_nbr": "delta_nbr",
    }
    raster_means = {}
    for key, scene_key in tensor_map.items():
        if scene_key not in scene:
            continue
        arr, _, _ = pad_stack(scene[scene_key])
        npy_path = PROC / f"{sid}_{key}.npy"
        paths[key] = save_npy(npy_path, arr)
        paths[f"{key}_tif"] = rel(scene[scene_key])
        raster_means[f"{key}_mean"] = array_nanmean(arr)

    csv_row = {}
    if not patch_csv.empty and "patch_id" in patch_csv.columns:
        hit = patch_csv[patch_csv["patch_id"].astype(str) == sid]
        if len(hit):
            csv_row = {k: v for k, v in hit.iloc[0].to_dict().items() if k not in GEO_COLS}

    records.append({
        "scene_id": sid,
        "native_shape": list(t0_rgb.shape[:2]),
        "vlm_size": list(TARGET_SIZE),
        "pad_meta": {k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()},
        "triplets_native": native_triplets,
        "triplets": vlm_triplets,
        "paths": paths,
        "raster_means": raster_means,
        "scalars": scalars.get(sid, {}),
        "csv_stats": csv_row,
        "sources_present": {
            "sentinel2_rgb": True,
            "sentinel2_optical10": "t0_optical" in scene,
            "sentinel1_vv_vh": "t0_s1" in scene,
            "chirps": "t0_rainfall" in scene,
            "era5_temp_sm": "t0_era5" in scene,
            "dem": "dem" in scene,
        },
    })
    print(f"{sid}: {len(vlm_triplets)} boxes | sources {records[-1]['sources_present']}")

TRIPLETS_JSON.write_text(json.dumps(records, indent=2, default=str))
print("wrote", TRIPLETS_JSON)
"""
    ),
    md("## 5. Sample triplet"),
    code(
        r"""sample = records[0]
print("scene", sample["scene_id"])
print("sources", sample["sources_present"])
print("sample triplet", sample["triplets"][0] if sample["triplets"] else None)
print("format: [[x_min, y_min, x_max, y_max], type, severity]")
"""
    ),
    md("## 6. Visual QA"),
    code(
        r"""fig, axes = plt.subplots(1, 4, figsize=(16, 4))
t0 = np.array(Image.open(ROOT / sample["paths"]["t0_image"]))
t1 = np.array(Image.open(ROOT / sample["paths"]["t1_image"]))
overlay = t1.copy()
for box, dtype, sev in sample["triplets"]:
    x0, y0, x1, y1 = box
    color = (255, 80, 40) if dtype == "Fire" else (40, 200, 80)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
    cv2.putText(overlay, f"{dtype}:{sev}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
ndvi_m = np.load(ROOT / sample["paths"]["delta_ndvi_mask"])
nbr_m = np.load(ROOT / sample["paths"]["delta_nbr_mask"])
axes[0].imshow(t0); axes[0].set_title("T0 RGB")
axes[1].imshow(overlay); axes[1].set_title("T1 + boxes")
axes[2].imshow(ndvi_m, cmap="gray"); axes[2].set_title("dNDVI Otsu")
axes[3].imshow(nbr_m, cmap="gray"); axes[3].set_title("dNBR Otsu")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
]

nb03 = [
    md(
        """# 03 — VLM JSONL (five-source evidence card)

Each line of `dataset.jsonl` is one patch:

- T0/T1 RGB (512×512)
- Sentinel-2 10-band optical + indices
- Sentinel-1 VV/VH
- CHIRPS rainfall
- ERA5-Land temperature & soil-moisture anomalies
- DEM elevation / slope / aspect
- Otsu triplets + explanation

**Lat/lon and place names are omitted** from the VLM-facing record (roadmap §5).
"""
    ),
    md("## 1. Load DIP records"),
    code(
        r"""from pathlib import Path
import json
import math
import sys

import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.text_utils import generate_vlm_explanation

TRIPLETS_JSON = ROOT / "data" / "raw_metadata" / "triplets.json"
OUT_JSONL = ROOT / "dataset.jsonl"
SIDECAR = ROOT / "data" / "raw_metadata" / "explanations_log.json"
FINAL_LABELS = ROOT.parent / "data" / "labels" / "final_labels.csv"

records = json.loads(TRIPLETS_JSON.read_text())
final_labels = pd.read_csv(FINAL_LABELS) if FINAL_LABELS.exists() else pd.DataFrame()
ref_map = dict(zip(final_labels.patch_id.astype(str), final_labels.reference_class)) if not final_labels.empty else {}
print("scenes", len(records), "| reference labels", len(ref_map))
"""
    ),
    md("## 2. Build evidence cards and generate explanations"),
    code(
        r"""def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if k not in {"lon", "lat"}}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def supporting_context(rec):
    s = rec.get("scalars") or {}
    csv = rec.get("csv_stats") or {}
    merged = {**csv, **s}
    keys = [
        "ndvi_t0", "ndvi_t1", "nbr_t0", "nbr_t1",
        "vv_t0", "vv_t1", "vh_t0", "vh_t1",
        "chirps_rainfall_t0", "chirps_rainfall_t1",
        "era5_temp_anomaly_t0", "era5_temp_anomaly_t1",
        "era5_sm_anomaly_t0", "era5_sm_anomaly_t1",
        "elevation_m", "slope_deg", "aspect_deg",
        "masked_frac_t0", "masked_frac_t1",
    ]
    # parent CSV used *_before / *_after names
    alias = {
        "ndvi_before": "ndvi_t0", "ndvi_after": "ndvi_t1",
        "nbr_before": "nbr_t0", "nbr_after": "nbr_t1",
        "vv_before": "vv_t0", "vv_after": "vv_t1",
        "vh_before": "vh_t0", "vh_after": "vh_t1",
        "chirps_rainfall_before": "chirps_rainfall_t0",
        "chirps_rainfall_after": "chirps_rainfall_t1",
        "era5_temp_anomaly_before": "era5_temp_anomaly_t0",
        "era5_temp_anomaly_after": "era5_temp_anomaly_t1",
        "era5_sm_anomaly_before": "era5_sm_anomaly_t0",
        "era5_sm_anomaly_after": "era5_sm_anomaly_t1",
        "masked_frac_before": "masked_frac_t0",
        "masked_frac_after": "masked_frac_t1",
    }
    out = {}
    for src, dst in alias.items():
        if src in merged and dst not in merged:
            merged[dst] = merged[src]
    for k in keys:
        if k in merged:
            out[k] = merged[k]
    return _clean(out)


OLLAMA_MODEL = "llama3"
logs = []
dataset_rows = []

for rec in records:
    ctx = supporting_context(rec)
    result = generate_vlm_explanation(rec.get("triplets") or [], model=OLLAMA_MODEL, context=ctx)
    logs.append({"scene_id": rec["scene_id"], "source": result["source"], "model": result["model"], "error": result["error"]})
    p = rec["paths"]
    s = rec.get("scalars") or {}
    csv = rec.get("csv_stats") or {}
    merged = {**csv, **s}

    row = {
        "id": rec["scene_id"],
        "task": "ground_forest_disturbance_in_spectral_evidence",
        "vlm_image_size": rec.get("vlm_size", [512, 512]),
        "images": {"t0": p.get("t0_image"), "t1": p.get("t1_image")},
        "sentinel2": {
            "bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
            "indices": ["NDVI", "EVI", "NDMI", "NBR"],
            "optical_t0": p.get("t0_optical"),
            "optical_t1": p.get("t1_optical"),
            "indices_t0": p.get("t0_indices"),
            "indices_t1": p.get("t1_indices"),
            "optical_t0_tif": p.get("t0_optical_tif"),
            "optical_t1_tif": p.get("t1_optical_tif"),
        },
        "sentinel1": {
            "polarizations": ["VV", "VH"],
            "t0": p.get("t0_s1"),
            "t1": p.get("t1_s1"),
            "t0_tif": p.get("t0_s1_tif"),
            "t1_tif": p.get("t1_s1_tif"),
            "orbit_pass": merged.get("s1_orbit_pass"),
            "n_scenes_t0": merged.get("s1_n_t0", merged.get("s1_n_before")),
            "n_scenes_t1": merged.get("s1_n_t1", merged.get("s1_n_after")),
        },
        "chirps": {
            "variable": "rainfall_mm",
            "t0": p.get("t0_rainfall"),
            "t1": p.get("t1_rainfall"),
            "t0_tif": p.get("t0_rainfall_tif"),
            "t1_tif": p.get("t1_rainfall_tif"),
            "rainfall_mm_t0": merged.get("chirps_rainfall_t0", merged.get("chirps_rainfall_before")),
            "rainfall_mm_t1": merged.get("chirps_rainfall_t1", merged.get("chirps_rainfall_after")),
        },
        "era5_land": {
            "variables": ["temperature_anomaly", "soil_moisture_anomaly"],
            "climatology": "2015-01-01/2021-12-31",
            "t0": p.get("t0_era5"),
            "t1": p.get("t1_era5"),
            "t0_tif": p.get("t0_era5_tif"),
            "t1_tif": p.get("t1_era5_tif"),
            "temp_anomaly_t0": merged.get("era5_temp_anomaly_t0", merged.get("era5_temp_anomaly_before")),
            "temp_anomaly_t1": merged.get("era5_temp_anomaly_t1", merged.get("era5_temp_anomaly_after")),
            "soil_moisture_anomaly_t0": merged.get("era5_sm_anomaly_t0", merged.get("era5_sm_anomaly_before")),
            "soil_moisture_anomaly_t1": merged.get("era5_sm_anomaly_t1", merged.get("era5_sm_anomaly_after")),
        },
        "dem": {
            "variables": ["elevation_m", "slope_deg", "aspect_deg"],
            "stack": p.get("dem"),
            "tif": p.get("dem_tif"),
            "elevation_m": merged.get("elevation_m"),
            "slope_deg": merged.get("slope_deg"),
            "aspect_deg": merged.get("aspect_deg"),
        },
        "evidence": {
            "delta_ndvi": p.get("delta_ndvi"),
            "delta_nbr": p.get("delta_nbr"),
            "delta_ndvi_tif": p.get("delta_ndvi_tif"),
            "delta_nbr_tif": p.get("delta_nbr_tif"),
            "delta_ndvi_mask": p.get("delta_ndvi_mask"),
            "delta_nbr_mask": p.get("delta_nbr_mask"),
        },
        "acquisition": {
            "season": "dry_season_jan_feb",
            "t0_window": [merged.get("t0_start", "2022-01-01"), merged.get("t0_end", "2022-02-28")],
            "t1_window": [merged.get("t1_start", "2023-01-01"), merged.get("t1_end", "2023-02-28")],
            "masked_fraction_t0": merged.get("masked_frac_t0", merged.get("masked_frac_before")),
            "masked_fraction_t1": merged.get("masked_frac_t1", merged.get("masked_frac_after")),
        },
        "triplets": [
            {"bbox": t[0], "disturbance_type": t[1], "severity": t[2]}
            for t in rec.get("triplets") or []
        ],
        "explanation": result["explanation"],
        "explanation_source": result["source"],
        "sources_present": rec.get("sources_present"),
        "candidate_class": merged.get("candidate_class"),
        "reference_class": ref_map.get(rec["scene_id"]),
    }
    dataset_rows.append(_clean(row))
    print(rec["scene_id"], result["source"], "boxes", len(row["triplets"]))

SIDECAR.write_text(json.dumps(logs, indent=2))
"""
    ),
    md("## 3. Write dataset.jsonl"),
    code(
        r"""with OUT_JSONL.open("w", encoding="utf-8") as f:
    for row in dataset_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Wrote {len(dataset_rows)} records -> {OUT_JSONL}")
preview = json.loads(OUT_JSONL.read_text().splitlines()[0])
preview["explanation"] = (preview.get("explanation") or "")[:240]
print(json.dumps({k: preview[k] for k in ["id", "sentinel2", "sentinel1", "chirps", "era5_land", "dem", "triplets", "explanation"] if k in preview}, indent=2)[:4000])
"""
    ),
    md("## 4. Completeness check"),
    code(
        r"""needed_blocks = ["images", "sentinel2", "sentinel1", "chirps", "era5_land", "dem", "evidence", "triplets", "explanation"]
s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
errors = []
for row in dataset_rows:
    for b in needed_blocks:
        if b not in row:
            errors.append((row["id"], b))
    if row["sentinel2"].get("bands") != s2_bands:
        errors.append((row["id"], "s2_band_list"))
    if row["sentinel1"].get("polarizations") != ["VV", "VH"]:
        errors.append((row["id"], "s1_pol"))
print("schema errors:", errors or "none")
print("records", len(dataset_rows))
print("ollama/fallback", {src: sum(1 for r in dataset_rows if r["explanation_source"] == src) for src in ("ollama", "fallback")})
"""
    ),
]

if __name__ == "__main__":
    write_nb("01_gee_extraction.ipynb", nb01)
    write_nb("02_dip_auto_classify.ipynb", nb02)
    write_nb("03_vlm_jsonl_builder.ipynb", nb03)
