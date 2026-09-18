#!/usr/bin/env python3
"""Convert S2 RGB GeoTIFFs to natural-color JPEG previews (fixes purple/wrong colormap).

Reads:  data/raw_tiffs/{patch}_T0_rgb.tif, {patch}_T1_rgb.tif
Writes: data/preview_chips/{patch}_before_rgb.jpg, {patch}_after_rgb.jpg

Also processes pilot_reference/ and outcasts/ if --all-folders.

Run from repo root:
  python scripts/build_rgb_previews.py
  python scripts/build_rgb_previews.py --all-folders
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "data" / "preview_chips"


def natural_rgb_uint8(arr: np.ndarray, vmax: float = 0.22, gamma: float = 1.0) -> np.ndarray:
    """Natural-color uint8 RGB via fixed Sentinel-2 reflectance window."""
    from src.dip_utils import percentile_stretch_rgb

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    return percentile_stretch_rgb(arr, vmax=vmax, gamma=gamma)


def convert_one(tif_path: Path, jpg_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        data = src.read()
    rgb = natural_rgb_uint8(data)
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    # Pillow via rasterio-free save
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(jpg_path, format="JPEG", quality=92)


def collect_jobs(folder: Path, out_dir: Path) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    for tif in sorted(folder.glob("*_T0_rgb.tif")):
        pid = tif.name.split("_T0_rgb.tif")[0]
        jobs.append((tif, out_dir / f"{pid}_before_rgb.jpg"))
    for tif in sorted(folder.glob("*_T1_rgb.tif")):
        pid = tif.name.split("_T1_rgb.tif")[0]
        jobs.append((tif, out_dir / f"{pid}_after_rgb.jpg"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-folders",
        action="store_true",
        help="Also convert pilot_reference/ and outcasts/ into subfolders.",
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw_tiffs")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    jobs = collect_jobs(args.input, args.output)
    if args.all_folders:
        for sub, name in [
            (ROOT / "data" / "pilot_reference", "pilot_reference"),
            (ROOT / "data" / "outcasts", "outcasts"),
        ]:
            if sub.exists():
                jobs.extend(collect_jobs(sub, args.output / name))

    if not jobs:
        print("No *_T0_rgb.tif / *_T1_rgb.tif files found.")
        return

    errors: list[str] = []
    for tif_path, jpg_path in tqdm(jobs, desc="RGB -> JPEG", unit="file"):
        if jpg_path.exists():
            continue
        try:
            convert_one(tif_path, jpg_path)
        except Exception as exc:
            errors.append(f"{tif_path.name}: {exc}")

    print(f"Wrote previews to {args.output.resolve()}")
    print(f"Total JPEGs: {len(list(args.output.rglob('*.jpg')))}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(" ", e)


if __name__ == "__main__":
    main()
