#!/usr/bin/env python3
"""Export five-source patch cubes from Google Earth Engine.

Modes (see configs/pilot_kanha.yaml):
  local            — geemap writes directly to data/raw_tiffs/  (default, fastest)
  drive            — queue EE Export.image.toDrive tasks
  local_then_drive — try local first, fall back to Drive per patch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ee  # noqa: E402

from src.drive_utils import poll_ee_tasks, save_export_manifest, start_drive_exports  # noqa: E402
from src.export_utils import build_export_images, inventory_patches  # noqa: E402
from src.gee_utils import (  # noqa: E402
    get_chirps_rainfall,
    get_dem_terrain,
    get_dominant_orbit_pass,
    get_era5_anomalies,
    get_s1_composite,
    get_s2_composite,
    patch_geometry,
    rgb_reflectance,
)
from src.paths import EXPORT_MANIFEST, RAW_META, RAW_TIFFS, SELECTED_PATCHES_CSV, ensure_data_dirs  # noqa: E402


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def export_local(specs, out_map: dict[str, Path]) -> None:
    import geemap

    for prefix, image, scale in specs:
        out_path = out_map[prefix]
        if out_path.exists():
            print("skip", out_path.name)
            continue
        geemap.ee_export_image(
            image,
            filename=str(out_path),
            scale=scale,
            region=out_map["_geom"],
            file_per_band=False,
        )
        print("wrote", out_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GEE patch cubes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_kanha.yaml")
    parser.add_argument("--mode", choices=["local", "drive", "local_then_drive"], default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    ensure_data_dirs()
    cfg = load_config(args.config)
    mode = args.mode or cfg["export"]["mode"]

    init_ee(cfg["project"]["gcp_project_id"])

    roi = ee.Geometry.Rectangle(cfg["study_area"]["roi_bbox"])
    t = cfg["temporal"]
    p = cfg["patch"]
    patch_size_m = p["patch_pixels"] * p["pixel_size_m"]

    s2_t0, _ = get_s2_composite(roi, t["before_start"], t["before_end"], p["max_cloud_pct"])
    s2_t1, _ = get_s2_composite(roi, t["after_start"], t["after_end"], p["max_cloud_pct"])
    orbit = get_dominant_orbit_pass(roi, t["before_start"], t["after_end"])
    s1_t0, _ = get_s1_composite(roi, t["before_start"], t["before_end"], orbit)
    s1_t1, _ = get_s1_composite(roi, t["after_start"], t["after_end"], orbit)
    chirps_t0, _ = get_chirps_rainfall(roi, t["before_start"], t["before_end"])
    chirps_t1, _ = get_chirps_rainfall(roi, t["after_start"], t["after_end"])
    era5_temp_t0, era5_sm_t0 = get_era5_anomalies(
        roi, t["before_start"], t["before_end"], t["climatology_start"], t["climatology_end"]
    )
    era5_temp_t1, era5_sm_t1 = get_era5_anomalies(
        roi, t["after_start"], t["after_end"], t["climatology_start"], t["climatology_end"]
    )
    era5_t0 = era5_temp_t0.addBands(era5_sm_t0)
    era5_t1 = era5_temp_t1.addBands(era5_sm_t1)
    dem = get_dem_terrain(roi)
    t0_rgb = rgb_reflectance(s2_t0)
    t1_rgb = rgb_reflectance(s2_t1)
    delta_ndvi = s2_t1.select("NDVI").subtract(s2_t0.select("NDVI")).rename("delta_ndvi")
    delta_nbr = s2_t1.select("NBR").subtract(s2_t0.select("NBR")).rename("delta_nbr")

    selected = pd.read_csv(SELECTED_PATCHES_CSV)
    if args.max_patches:
        selected = selected.head(args.max_patches)

    drive_folder = cfg["export"]["drive_folder"]
    manifest: list[dict] = []
    all_tasks = []

    for _, row in selected.iterrows():
        pid = str(row["patch_id"])
        geom = patch_geometry(float(row["lon"]), float(row["lat"]), patch_size_m, roi)
        specs = build_export_images(
            pid,
            geom,
            s2_t0=s2_t0,
            s2_t1=s2_t1,
            s1_t0=s1_t0,
            s1_t1=s1_t1,
            chirps_t0=chirps_t0,
            chirps_t1=chirps_t1,
            era5_t0=era5_t0,
            era5_t1=era5_t1,
            dem=dem,
            t0_rgb=t0_rgb,
            t1_rgb=t1_rgb,
            delta_ndvi=delta_ndvi,
            delta_nbr=delta_nbr,
            pixel_size_m=p["pixel_size_m"],
        )

        out_map = {prefix: RAW_TIFFS / f"{prefix}.tif" for prefix, _, _ in specs}
        out_map["_geom"] = geom

        if mode in ("local", "local_then_drive"):
            try:
                export_local(specs, out_map)
                manifest.append({"patch_id": pid, "method": "local", "status": "ok"})
                continue
            except Exception as exc:
                if mode == "local":
                    manifest.append({"patch_id": pid, "method": "local", "status": "failed", "error": str(exc)})
                    print(f"FAILED local {pid}: {exc}")
                    continue
                print(f"local failed {pid}: {exc} -> Drive")

        tasks = start_drive_exports(specs, drive_folder=drive_folder, geom=geom)
        all_tasks.extend(tasks)
        manifest.append({"patch_id": pid, "method": "drive", "tasks": len(tasks)})

    if all_tasks:
        print(f"Polling {len(all_tasks)} Drive export tasks...")
        states = poll_ee_tasks(all_tasks)
        failed = [k for k, v in states.items() if v != "COMPLETED"]
        print("Drive task states:", states)
        if failed:
            print("WARNING: some Drive tasks did not complete. Run scripts/02_download_drive.py after they finish.")

    save_export_manifest(EXPORT_MANIFEST, manifest)
    inv = inventory_patches(RAW_TIFFS, selected["patch_id"].astype(str))
    complete = sum(1 for m in inv.values() if m["complete"])
    print(f"Export done. Complete patches: {complete}/{len(inv)}")
    print(f"Manifest: {EXPORT_MANIFEST}")


if __name__ == "__main__":
    main()
