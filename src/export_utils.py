"""Export specifications, inventory, and QA for the five-source patch cube."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# 15 GeoTIFFs per patch (roadmap-aligned five-source cube)
PATCH_SUFFIXES: tuple[str, ...] = (
    "_T0_rgb",
    "_T1_rgb",
    "_T0_s2_optical",
    "_T1_s2_optical",
    "_T0_indices",
    "_T1_indices",
    "_T0_s1",
    "_T1_s1",
    "_T0_rainfall",
    "_T1_rainfall",
    "_T0_era5",
    "_T1_era5",
    "_dem",
    "_delta_ndvi",
    "_delta_nbr",
)


def patch_tif_path(raw_dir: Path, patch_id: str, suffix: str) -> Path:
    return raw_dir / f"{patch_id}{suffix}.tif"


def missing_suffixes(raw_dir: Path, patch_id: str) -> list[str]:
    return [s for s in PATCH_SUFFIXES if not patch_tif_path(raw_dir, patch_id, s).exists()]


def patch_is_complete(raw_dir: Path, patch_id: str) -> bool:
    return len(missing_suffixes(raw_dir, patch_id)) == 0


def inventory_patches(raw_dir: Path, patch_ids: Iterable[str]) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for pid in patch_ids:
        missing = missing_suffixes(raw_dir, pid)
        report[pid] = {
            "complete": not missing,
            "present": len(PATCH_SUFFIXES) - len(missing),
            "expected": len(PATCH_SUFFIXES),
            "missing": missing,
        }
    return report


def organize_inbox_tiffs(inbox_dir: Path, raw_dir: Path) -> list[Path]:
    """Move/copy *.tif from Drive download inbox into data/raw_tiffs/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for src in sorted(inbox_dir.glob("*.tif")):
        dst = raw_dir / src.name
        if dst.exists():
            continue
        dst.write_bytes(src.read_bytes())
        moved.append(dst)
    return moved


def build_export_images(
    patch_id: str,
    geom,
    *,
    s2_t0,
    s2_t1,
    s1_t0,
    s1_t1,
    chirps_t0,
    chirps_t1,
    era5_t0,
    era5_t1,
    dem,
    t0_rgb,
    t1_rgb,
    delta_ndvi,
    delta_nbr,
    pixel_size_m: int,
):
    """Return (drive_prefix, image, scale) tuples for one patch. Requires ee at call time."""
    import ee  # noqa: WPS433 — optional dependency for export scripts only

    from .gee_utils import CHIRPS_SCALE, DEM_SCALE, ERA5_SCALE, S2_INDICES, S2_OPTICAL_BANDS

    return [
        (f"{patch_id}_T0_rgb", t0_rgb.clip(geom), pixel_size_m),
        (f"{patch_id}_T1_rgb", t1_rgb.clip(geom), pixel_size_m),
        (f"{patch_id}_T0_s2_optical", s2_t0.select(S2_OPTICAL_BANDS).clip(geom), pixel_size_m),
        (f"{patch_id}_T1_s2_optical", s2_t1.select(S2_OPTICAL_BANDS).clip(geom), pixel_size_m),
        (f"{patch_id}_T0_indices", s2_t0.select(S2_INDICES).clip(geom), pixel_size_m),
        (f"{patch_id}_T1_indices", s2_t1.select(S2_INDICES).clip(geom), pixel_size_m),
        (f"{patch_id}_T0_s1", s1_t0.select(["VV", "VH"]).clip(geom), pixel_size_m),
        (f"{patch_id}_T1_s1", s1_t1.select(["VV", "VH"]).clip(geom), pixel_size_m),
        (f"{patch_id}_T0_rainfall", chirps_t0.clip(geom), CHIRPS_SCALE),
        (f"{patch_id}_T1_rainfall", chirps_t1.clip(geom), CHIRPS_SCALE),
        (f"{patch_id}_T0_era5", era5_t0.clip(geom), ERA5_SCALE),
        (f"{patch_id}_T1_era5", era5_t1.clip(geom), ERA5_SCALE),
        (f"{patch_id}_dem", dem.clip(geom), DEM_SCALE),
        (f"{patch_id}_delta_ndvi", delta_ndvi.clip(geom), pixel_size_m),
        (f"{patch_id}_delta_nbr", delta_nbr.clip(geom), pixel_size_m),
    ]
