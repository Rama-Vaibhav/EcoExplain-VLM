#!/usr/bin/env python3
"""Run DIP processing + VLM JSONL builder (notebooks 02 + 03 as a script)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_cards import write_evidence_cards  # noqa: E402
from src.dip_utils import (  # noqa: E402
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
from src.paths import (  # noqa: E402
    ANNOTATIONS,
    DATASET_JSONL,
    EXPLANATIONS_LOG,
    MASKS,
    PATCH_SCALARS_JSON,
    PROCESSED,
    RAW_META,
    RAW_TIFFS,
    SELECTED_PATCHES_CSV,
    TRIPLETS_JSON,
    ensure_data_dirs,
)
from src.text_utils import generate_vlm_explanation  # noqa: E402


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _clean(obj):
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


def supporting_context(rec, merged: dict) -> dict:
    keys = [
        "ndvi_t0", "ndvi_t1", "nbr_t0", "nbr_t1",
        "vv_t0", "vv_t1", "vh_t0", "vh_t1",
        "chirps_rainfall_t0", "chirps_rainfall_t1",
        "era5_temp_anomaly_t0", "era5_temp_anomaly_t1",
        "era5_sm_anomaly_t0", "era5_sm_anomaly_t1",
        "elevation_m", "slope_deg", "aspect_deg",
        "masked_frac_t0", "masked_frac_t1",
    ]
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
    for src, dst in alias.items():
        if src in merged and dst not in merged:
            merged[dst] = merged[src]
    return _clean({k: merged[k] for k in keys if k in merged})


def run_dip(target_size: tuple[int, int]) -> list[dict]:
    patch_csv = pd.read_csv(SELECTED_PATCHES_CSV) if SELECTED_PATCHES_CSV.exists() else pd.DataFrame()
    scalars = json.loads(PATCH_SCALARS_JSON.read_text()) if PATCH_SCALARS_JSON.exists() else {}
    scenes = discover_scenes(RAW_TIFFS)
    if not scenes:
        raise FileNotFoundError(f"No complete scenes in {RAW_TIFFS}. Run scripts/01_export_gee.py first.")

    records = []
    geo_cols = {"lon", "lat", "patch_id"}

    def pad_stack(path, native_boxes=None):
        hwc = load_geotiff_hwc(path)
        if hwc.shape[-1] == 1:
            hwc = hwc[..., 0]
        out, boxes, meta = pad_and_resize(hwc, target_size, bboxes=native_boxes, keep_dtype=True)
        return out.astype(np.float32), boxes, meta

    for scene in scenes:
        sid = scene["scene_id"]
        if sid == "demo_kanha":
            continue

        ndvi_mask = process_tensor_otsu(scene["delta_ndvi"])
        nbr_mask = process_tensor_otsu(scene["delta_nbr"])
        delta_ndvi = read_single_band(scene["delta_ndvi"])
        delta_nbr = read_single_band(scene["delta_nbr"])
        native_triplets = build_disturbance_triplets(ndvi_mask, nbr_mask, delta_ndvi, delta_nbr)
        native_boxes = [t[0] for t in native_triplets]

        t0_rgb = percentile_stretch_rgb(load_rgb_geotiff(scene["t0_rgb"]))
        t1_rgb = percentile_stretch_rgb(load_rgb_geotiff(scene["t1_rgb"]))
        t0_512, boxes_512, meta = pad_and_resize(t0_rgb, target_size, bboxes=native_boxes)
        t1_512, _, _ = pad_and_resize(t1_rgb, target_size, bboxes=native_boxes)
        vlm_triplets = [[b, t[1], t[2]] for b, t in zip(boxes_512, native_triplets)]

        t0_jpg = PROCESSED / f"{sid}_T0.jpg"
        t1_jpg = PROCESSED / f"{sid}_T1.jpg"
        Image.fromarray(t0_512).save(t0_jpg, quality=95)
        Image.fromarray(t1_512).save(t1_jpg, quality=95)

        ndvi_512, _, _ = pad_and_resize((ndvi_mask * 255).astype(np.uint8), target_size)
        nbr_512, _, _ = pad_and_resize((nbr_mask * 255).astype(np.uint8), target_size)
        np.save(MASKS / f"{sid}_delta_ndvi_mask.npy", ndvi_512)
        np.save(MASKS / f"{sid}_delta_nbr_mask.npy", nbr_512)

        paths = {
            "t0_image": rel(t0_jpg),
            "t1_image": rel(t1_jpg),
            "delta_ndvi_mask": rel(MASKS / f"{sid}_delta_ndvi_mask.npy"),
            "delta_nbr_mask": rel(MASKS / f"{sid}_delta_nbr_mask.npy"),
        }
        tensor_map = {
            "t0_optical": "t0_optical", "t1_optical": "t1_optical",
            "t0_indices": "t0_indices", "t1_indices": "t1_indices",
            "t0_s1": "t0_s1", "t1_s1": "t1_s1",
            "t0_rainfall": "t0_rainfall", "t1_rainfall": "t1_rainfall",
            "t0_era5": "t0_era5", "t1_era5": "t1_era5",
            "dem": "dem",
            "delta_ndvi": "delta_ndvi", "delta_nbr": "delta_nbr",
        }
        raster_means = {}
        for key, scene_key in tensor_map.items():
            if scene_key not in scene:
                continue
            arr, _, _ = pad_stack(scene[scene_key])
            npy_path = PROCESSED / f"{sid}_{key}.npy"
            np.save(npy_path, arr)
            paths[key] = rel(npy_path)
            paths[f"{key}_tif"] = rel(scene[scene_key])
            raster_means[f"{key}_mean"] = array_nanmean(arr)

        csv_row = {}
        if not patch_csv.empty and "patch_id" in patch_csv.columns:
            hit = patch_csv[patch_csv["patch_id"].astype(str) == sid]
            if len(hit):
                csv_row = {k: v for k, v in hit.iloc[0].to_dict().items() if k not in geo_cols}

        records.append({
            "scene_id": sid,
            "native_shape": list(t0_rgb.shape[:2]),
            "vlm_size": list(target_size),
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
        print(f"{sid}: {len(vlm_triplets)} boxes")

    TRIPLETS_JSON.write_text(json.dumps(records, indent=2, default=str))
    return records


def build_jsonl(records: list[dict], model: str, exclude_demo: bool) -> tuple[list[dict], list[dict]]:
    logs = []
    rows = []
    for rec in records:
        if exclude_demo and rec["scene_id"] == "demo_kanha":
            continue
        merged = {**(rec.get("csv_stats") or {}), **(rec.get("scalars") or {})}
        ctx = supporting_context(rec, merged)
        result = generate_vlm_explanation(rec.get("triplets") or [], model=model, context=ctx)
        logs.append({
            "scene_id": rec["scene_id"],
            "source": result["source"],
            "model": result["model"],
            "error": result["error"],
        })
        p = rec["paths"]
        row = {
            "id": rec["scene_id"],
            "task": "ground_forest_disturbance_in_spectral_evidence",
            "vlm_image_size": rec.get("vlm_size", [512, 512]),
            "images": {"t0": p.get("t0_image"), "t1": p.get("t1_image")},
            "sentinel2": {
                "bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
                "indices": ["NDVI", "EVI", "NDMI", "NBR"],
                "optical_t0": p.get("t0_optical"), "optical_t1": p.get("t1_optical"),
                "indices_t0": p.get("t0_indices"), "indices_t1": p.get("t1_indices"),
            },
            "sentinel1": {
                "polarizations": ["VV", "VH"],
                "t0": p.get("t0_s1"), "t1": p.get("t1_s1"),
                "orbit_pass": merged.get("s1_orbit_pass"),
            },
            "chirps": {
                "variable": "rainfall_mm",
                "t0": p.get("t0_rainfall"), "t1": p.get("t1_rainfall"),
                "rainfall_mm_t0": merged.get("chirps_rainfall_t0", merged.get("chirps_rainfall_before")),
                "rainfall_mm_t1": merged.get("chirps_rainfall_t1", merged.get("chirps_rainfall_after")),
            },
            "era5_land": {
                "variables": ["temperature_anomaly", "soil_moisture_anomaly"],
                "climatology": "2015-01-01/2021-12-31",
                "t0": p.get("t0_era5"), "t1": p.get("t1_era5"),
                "temp_anomaly_t0": merged.get("era5_temp_anomaly_t0", merged.get("era5_temp_anomaly_before")),
                "temp_anomaly_t1": merged.get("era5_temp_anomaly_t1", merged.get("era5_temp_anomaly_after")),
                "soil_moisture_anomaly_t0": merged.get("era5_sm_anomaly_t0", merged.get("era5_sm_anomaly_before")),
                "soil_moisture_anomaly_t1": merged.get("era5_sm_anomaly_t1", merged.get("era5_sm_anomaly_after")),
            },
            "dem": {
                "variables": ["elevation_m", "slope_deg", "aspect_deg"],
                "stack": p.get("dem"),
                "elevation_m": merged.get("elevation_m"),
                "slope_deg": merged.get("slope_deg"),
                "aspect_deg": merged.get("aspect_deg"),
            },
            "evidence": {
                "delta_ndvi": p.get("delta_ndvi"), "delta_nbr": p.get("delta_nbr"),
                "delta_ndvi_mask": p.get("delta_ndvi_mask"), "delta_nbr_mask": p.get("delta_nbr_mask"),
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
        }
        rows.append(_clean(row))
    return rows, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_kanha.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    ensure_data_dirs()

    target_size = tuple(cfg["vlm"]["image_size"])
    records = run_dip(target_size)
    rows, logs = build_jsonl(records, cfg["vlm"]["ollama_model"], cfg["vlm"]["exclude_demo_from_jsonl"])

    with DATASET_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    EXPLANATIONS_LOG.write_text(json.dumps(logs, indent=2))
    explanations = {log["scene_id"]: {"explanation": next(r["explanation"] for r in rows if r["id"] == log["scene_id"]), "source": log["source"]} for log in logs}
    write_evidence_cards(records, ANNOTATIONS, explanations)

    print(f"Wrote {len(rows)} records -> {DATASET_JSONL}")
    print(f"Evidence cards -> {ANNOTATIONS}")


if __name__ == "__main__":
    main()
