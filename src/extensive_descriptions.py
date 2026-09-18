"""Per-patch raster analysis and evidence-driven text (not template rephrasing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .dip_utils import load_rgb_geotiff, percentile_stretch_rgb

INDEX_NAMES = ["NDVI", "EVI", "NDMI", "NBR"]
ANALYSIS_VERSION = 2


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _safe_mean(arr: Optional[np.ndarray]) -> Optional[float]:
    if arr is None or arr.size == 0:
        return None
    v = _finite(arr.astype(np.float64))
    return float(v.mean()) if v.size else None


def _safe_std(arr: Optional[np.ndarray]) -> Optional[float]:
    if arr is None or arr.size == 0:
        return None
    v = _finite(arr.astype(np.float64))
    return float(v.std()) if v.size > 1 else 0.0


def _load_npy(path: Optional[str], root: Path) -> Optional[np.ndarray]:
    if not path:
        return None
    p = root / path
    return np.load(p) if p.exists() else None


def _mask_fraction(m: Optional[np.ndarray]) -> float:
    if m is None:
        return 0.0
    x = m.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return float(np.clip(x, 0, 1).mean())


def _delta_raster_stats(arr: Optional[np.ndarray]) -> dict[str, Optional[float]]:
    if arr is None:
        return {}
    if arr.ndim == 3:
        arr = arr[..., 0]
    v = _finite(arr.astype(np.float64))
    if v.size == 0:
        return {}
    return {
        "mean": round(float(v.mean()), 5),
        "std": round(float(v.std()), 5),
        "min": round(float(v.min()), 5),
        "max": round(float(v.max()), 5),
        "p10": round(float(np.percentile(v, 10)), 5),
        "p90": round(float(np.percentile(v, 90)), 5),
        "fraction_below_-0.05": round(float((v < -0.05).mean()), 4),
        "fraction_above_+0.05": round(float((v > 0.05).mean()), 4),
    }


def _terrain_rgb_stats(tif_path: Path) -> dict[str, Any]:
    if not tif_path.exists():
        return {}
    rgb_u8 = percentile_stretch_rgb(load_rgb_geotiff(tif_path))
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return {
        "mean_r": round(float(r.mean()), 4),
        "mean_g": round(float(g.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "brightness_mean": round(float(lum.mean()), 4),
        "brightness_std": round(float(lum.std()), 4),
        "green_dominant_fraction": round(float(((g > r) & (g > b)).mean()), 4),
        "native_shape": list(rgb.shape[:2]),
    }


def _index_means(arr: Optional[np.ndarray]) -> dict[str, Optional[float]]:
    if arr is None or arr.ndim < 3:
        return {n: None for n in INDEX_NAMES}
    out: dict[str, Optional[float]] = {}
    for i, name in enumerate(INDEX_NAMES):
        if i < arr.shape[-1]:
            m = _safe_mean(arr[..., i])
            out[name] = round(m, 4) if m is not None else None
        else:
            out[name] = None
    return out


def _index_deltas(t0: dict[str, Optional[float]], t1: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for name in INDEX_NAMES:
        a, b = t0.get(name), t1.get(name)
        out[name] = round(b - a, 4) if a is not None and b is not None else None
    return out


def _box_area_fraction(box: list, canvas: int = 512) -> float:
    x0, y0, x1, y1 = box
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return round(area / (canvas * canvas), 4)


def _infer_signals(analysis: dict[str, Any]) -> dict[str, Any]:
    """Rule-based interpretation from measured rasters (not wording templates)."""
    d_ndvi = analysis["index_delta"].get("NDVI")
    d_nbr = analysis["index_delta"].get("NBR")
    d_evi = analysis["index_delta"].get("EVI")
    cand = analysis.get("candidate_class") or "unknown"

    ndvi_drop = analysis["delta_ndvi_raster"].get("fraction_below_-0.05", 0) or 0
    ndvi_gain = analysis["delta_ndvi_raster"].get("fraction_above_+0.05", 0) or 0
    mask_ndvi = analysis["otsu_mask_fraction_ndvi"]
    mask_nbr = analysis["otsu_mask_fraction_nbr"]
    n_boxes = analysis["disturbance_box_count"]

    # Vegetation trend from indices + delta raster
    if d_ndvi is not None and d_ndvi <= -0.04 and ndvi_drop >= 0.08:
        veg_trend = "strong_decline"
    elif d_ndvi is not None and d_ndvi <= -0.02:
        veg_trend = "moderate_decline"
    elif d_ndvi is not None and d_ndvi >= 0.02 and ndvi_gain >= 0.08:
        veg_trend = "greening"
    elif d_ndvi is not None and d_ndvi >= 0.01:
        veg_trend = "slight_greening"
    else:
        veg_trend = "stable"

    if mask_nbr > mask_ndvi + 0.02 and (d_nbr or 0) < -0.02:
        fire_likelihood = "elevated_burn_signal"
    elif mask_ndvi > 0.02 and n_boxes > 0:
        fire_likelihood = "deforestation_pattern"
    else:
        fire_likelihood = "low"

    # Consistency between sampling label and measured spectra
    if cand == "decline" and veg_trend in ("strong_decline", "moderate_decline"):
        label_consistency = "agrees"
    elif cand == "increase" and veg_trend in ("greening", "slight_greening"):
        label_consistency = "agrees"
    elif cand == "stable" and veg_trend == "stable" and n_boxes == 0:
        label_consistency = "agrees"
    elif cand == "stable" and n_boxes > 0:
        label_consistency = "spatial_anomaly"
    else:
        label_consistency = "partial_mismatch"

    sar_vv_d = analysis.get("sar_vv_delta")
    if sar_vv_d is not None and abs(sar_vv_d) >= 0.8:
        sar_role = "notable_backscatter_shift"
    else:
        sar_role = "secondary"

    rain_d = analysis.get("rainfall_delta_mm")
    if rain_d is not None and rain_d <= -25:
        climate_role = "drier_second_window"
    elif rain_d is not None and rain_d >= 25:
        climate_role = "wetter_second_window"
    else:
        climate_role = "neutral"

    slope = analysis.get("slope_deg") or 0
    terrain_role = "steep_terrain" if slope >= 12 else "gentle_terrain"

    return {
        "vegetation_trend": veg_trend,
        "fire_burn_likelihood": fire_likelihood,
        "candidate_class_consistency": label_consistency,
        "sar_evidence_strength": sar_role,
        "climate_evidence_strength": climate_role,
        "terrain_context": terrain_role,
        "primary_spectral_drivers": {
            "ndvi_delta": d_ndvi,
            "nbr_delta": d_nbr,
            "evi_delta": d_evi,
        },
    }


def analyze_patch_files(row: dict[str, Any], root: Path, csv_row: Optional[dict] = None) -> dict[str, Any]:
    pid = row["id"]
    tif_dir = root / "data" / "raw_tiffs"

    t0_rgb = _terrain_rgb_stats(tif_dir / f"{pid}_T0_rgb.tif")
    t1_rgb = _terrain_rgb_stats(tif_dir / f"{pid}_T1_rgb.tif")

    idx_t0 = _load_npy(row.get("sentinel2", {}).get("indices_t0"), root)
    idx_t1 = _load_npy(row.get("sentinel2", {}).get("indices_t1"), root)
    indices_t0 = _index_means(idx_t0)
    indices_t1 = _index_means(idx_t1)
    index_delta = _index_deltas(indices_t0, indices_t1)

    s1_t0 = _load_npy(row.get("sentinel1", {}).get("t0"), root)
    s1_t1 = _load_npy(row.get("sentinel1", {}).get("t1"), root)

    d_ndvi = _load_npy(row.get("evidence", {}).get("delta_ndvi"), root)
    d_nbr = _load_npy(row.get("evidence", {}).get("delta_nbr"), root)
    m_ndvi = _load_npy(row.get("evidence", {}).get("delta_ndvi_mask"), root)
    m_nbr = _load_npy(row.get("evidence", {}).get("delta_nbr_mask"), root)

    triplets = row.get("triplets") or []
    box_areas = [_box_area_fraction(tr["bbox"]) for tr in triplets]

    chirps = row.get("chirps") or {}
    era5 = row.get("era5_land") or {}
    dem = row.get("dem") or {}

    vv_t0 = _safe_mean(s1_t0[..., 0]) if s1_t0 is not None and s1_t0.ndim >= 2 else None
    vv_t1 = _safe_mean(s1_t1[..., 0]) if s1_t1 is not None and s1_t1.ndim >= 2 else None
    vh_t0 = _safe_mean(s1_t0[..., 1]) if s1_t0 is not None and s1_t0.shape[-1] >= 2 else None
    vh_t1 = _safe_mean(s1_t1[..., 1]) if s1_t1 is not None and s1_t1.shape[-1] >= 2 else None

    csv_row = csv_row or {}
    analysis: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "patch_id": pid,
        "files_read": {
            "t0_rgb_tif": (tif_dir / f"{pid}_T0_rgb.tif").exists(),
            "t1_rgb_tif": (tif_dir / f"{pid}_T1_rgb.tif").exists(),
            "indices_t0_npy": idx_t0 is not None,
            "indices_t1_npy": idx_t1 is not None,
            "delta_ndvi_npy": d_ndvi is not None,
            "delta_nbr_npy": d_nbr is not None,
            "s1_t0_npy": s1_t0 is not None,
            "s1_t1_npy": s1_t1 is not None,
        },
        "terrain_rgb_t0": t0_rgb,
        "terrain_rgb_t1": t1_rgb,
        "brightness_delta": (
            round(t1_rgb["brightness_mean"] - t0_rgb["brightness_mean"], 4)
            if t0_rgb and t1_rgb
            else None
        ),
        "indices_t0": indices_t0,
        "indices_t1": indices_t1,
        "index_delta": index_delta,
        "delta_ndvi_raster": _delta_raster_stats(d_ndvi),
        "delta_nbr_raster": _delta_raster_stats(d_nbr),
        "otsu_mask_fraction_ndvi": round(_mask_fraction(m_ndvi), 4),
        "otsu_mask_fraction_nbr": round(_mask_fraction(m_nbr), 4),
        "sar_vv_mean_t0": round(vv_t0, 3) if vv_t0 is not None else None,
        "sar_vv_mean_t1": round(vv_t1, 3) if vv_t1 is not None else None,
        "sar_vh_mean_t0": round(vh_t0, 3) if vh_t0 is not None else None,
        "sar_vh_mean_t1": round(vh_t1, 3) if vh_t1 is not None else None,
        "sar_vv_delta": round(vv_t1 - vv_t0, 3) if vv_t0 is not None and vv_t1 is not None else None,
        "sar_vh_delta": round(vh_t1 - vh_t0, 3) if vh_t0 is not None and vh_t1 is not None else None,
        "rainfall_mm_t0": chirps.get("rainfall_mm_t0"),
        "rainfall_mm_t1": chirps.get("rainfall_mm_t1"),
        "rainfall_delta_mm": (
            round(chirps["rainfall_mm_t1"] - chirps["rainfall_mm_t0"], 2)
            if chirps.get("rainfall_mm_t0") is not None and chirps.get("rainfall_mm_t1") is not None
            else None
        ),
        "temp_anomaly_t0": era5.get("temp_anomaly_t0"),
        "temp_anomaly_t1": era5.get("temp_anomaly_t1"),
        "soil_moisture_anomaly_t0": era5.get("soil_moisture_anomaly_t0"),
        "soil_moisture_anomaly_t1": era5.get("soil_moisture_anomaly_t1"),
        "elevation_m": dem.get("elevation_m"),
        "slope_deg": dem.get("slope_deg"),
        "aspect_deg": dem.get("aspect_deg"),
        "disturbance_box_count": len(triplets),
        "disturbance_box_area_fractions": box_areas,
        "disturbance_box_total_area_fraction": round(sum(box_areas), 4) if box_areas else 0.0,
        "candidate_class": row.get("candidate_class"),
        "csv_ndvi_change": csv_row.get("ndvi_change"),
        "csv_nbr_change": csv_row.get("nbr_change"),
        "csv_masked_frac_t0": csv_row.get("masked_frac_before") or csv_row.get("masked_frac_t0"),
    }

    analysis["inference"] = _infer_signals(analysis)
    return analysis


def _describe_terrain(stats: dict[str, Any], label: str) -> str:
    if not stats:
        return f"{label}: RGB GeoTIFF missing."
    br, gd, sh = stats["brightness_mean"], stats["green_dominant_fraction"], stats.get("brightness_std", 0)
    texture = "high internal contrast" if sh > 0.12 else "uniform illumination"
    cover = (
        "canopy-green dominates"
        if gd > 0.5
        else "mixed canopy and non-photosynthetic pixels"
        if gd > 0.3
        else "open or shadowed surfaces are frequent"
    )
    return (
        f"{label} terrain RGB ({stats['native_shape'][0]}×{stats['native_shape'][1]} native, "
        f"stretched for display): brightness={br:.3f}, green-cover={gd:.2f}, {texture}; {cover}."
    )


def synthesize_narrative(row: dict[str, Any], analysis: dict[str, Any]) -> dict[str, str]:
    """Build text only from analysis fields and inference — no rotating openers."""
    pid = analysis["patch_id"]
    inf = analysis["inference"]
    it0, it1, idel = analysis["indices_t0"], analysis["indices_t1"], analysis["index_delta"]
    dnd = analysis["delta_ndvi_raster"]
    dnb = analysis["delta_nbr_raster"]

    t0_line = _describe_terrain(analysis["terrain_rgb_t0"], "T0 (2022)")
    t1_line = _describe_terrain(analysis["terrain_rgb_t1"], "T1 (2023)")
    bd = analysis.get("brightness_delta")
    visual_pair = (
        f"{t0_line} {t1_line} "
        f"Brightness change T1−T0={bd:+.4f} on natural-color chips."
        if bd is not None
        else f"{t0_line} {t1_line}"
    )

    spectral = (
        f"On the 512×512 aligned stack, mean NDVI {it0.get('NDVI')}→{it1.get('NDVI')} "
        f"(Δ={idel.get('NDVI'):+.4f}), NBR {it0.get('NBR')}→{it1.get('NBR')} "
        f"(Δ={idel.get('NBR'):+.4f}), EVI Δ={idel.get('EVI'):+.4f}, NDMI Δ={idel.get('NDMI'):+.4f}. "
        f"ΔNDVI raster: mean={dnd.get('mean')}, {dnd.get('fraction_below_-0.05', 0)*100:.1f}% pixels below −0.05, "
        f"{dnd.get('fraction_above_+0.05', 0)*100:.1f}% above +0.05; "
        f"ΔNBR raster mean={dnb.get('mean')}."
    )

    sar = (
        f"SAR (512×512 means): VV {analysis['sar_vv_mean_t0']}→{analysis['sar_vv_mean_t1']} dB "
        f"(Δ{analysis['sar_vv_delta']:+.2f}), VH {analysis['sar_vh_mean_t0']}→{analysis['sar_vh_mean_t1']} dB "
        f"(Δ{analysis['sar_vh_delta']:+.2f}); interpreted as {inf['sar_evidence_strength']}."
        if analysis["sar_vv_delta"] is not None
        else "SAR tensors missing or empty."
    )

    if analysis.get("rainfall_mm_t0") is not None and analysis.get("rainfall_mm_t1") is not None:
        climate = (
            f"CHIRPS Jan–Feb total {analysis['rainfall_mm_t0']:.1f}→{analysis['rainfall_mm_t1']:.1f} mm "
            f"(Δ{analysis['rainfall_delta_mm']:+.1f} mm, {inf['climate_evidence_strength']}). "
            f"ERA5-Land anomalies: temperature {analysis['temp_anomaly_t0']:+.2f}→{analysis['temp_anomaly_t1']:+.2f} °C, "
            f"soil moisture {analysis['soil_moisture_anomaly_t0']:+.3f}→{analysis['soil_moisture_anomaly_t1']:+.3f} "
            f"vs 2015–2021 climatology."
        )
    else:
        climate = "Climate scalars unavailable for this record."

    terrain = (
        f"NASADEM at patch center: {analysis['elevation_m']:.0f} m, slope {analysis['slope_deg']:.1f}°, "
        f"aspect {analysis['aspect_deg']:.0f}° ({inf['terrain_context']})."
    )

    boxes = row.get("triplets") or []
    if boxes:
        parts = []
        for i, tr in enumerate(boxes):
            b = tr["bbox"]
            af = analysis["disturbance_box_area_fractions"][i]
            parts.append(
                f"{tr['disturbance_type']} {tr['severity']} at [{b[0]},{b[1]},{b[2]},{b[3]}] "
                f"({af*100:.1f}% of canvas)"
            )
        spatial = (
            f"Otsu segmentation produced {len(boxes)} region(s): "
            + "; ".join(parts)
            + f". Combined boxed area ≈{analysis['disturbance_box_total_area_fraction']*100:.1f}%."
        )
    else:
        spatial = (
            f"No Otsu boxes (NDVI mask active on {analysis['otsu_mask_fraction_ndvi']*100:.1f}% of pixels, "
            f"NBR mask on {analysis['otsu_mask_fraction_nbr']*100:.1f}%)."
        )

    conclusion = (
        f"Inference for {pid}: vegetation_trend={inf['vegetation_trend']}, "
        f"burn/deforestation cue={inf['fire_burn_likelihood']}, "
        f"candidate_class={analysis['candidate_class']} ({inf['candidate_class_consistency']} with spectra/boxes)."
    )

    # Lead paragraph follows strongest measured signal
    lead_map = {
        "strong_decline": f"Spectral loss dominates {pid}: ",
        "moderate_decline": f"Moderate canopy loss is measured for {pid}: ",
        "greening": f"Greening signal dominates {pid}: ",
        "slight_greening": f"Slight greening is measured for {pid}: ",
        "stable": f"Spectral stability dominates {pid}: ",
    }
    lead = lead_map.get(inf["vegetation_trend"], f"Summary for {pid}: ")

    analyst = " ".join([lead, spectral, spatial, visual_pair, sar, climate, terrain, conclusion])

    user_prompt = (
        f"Task: {row.get('task')}\n"
        f"Condition C evidence (measured on disk, analysis v{ANALYSIS_VERSION}):\n"
        f"{visual_pair}\n{spectral}\n{sar}\n{climate}\n{terrain}\n{spatial}\n{conclusion}\n"
        f"Respond with class (stable/decline/increase) and a short grounded explanation."
    )

    return {
        "visual_description_t0": t0_line,
        "visual_description_t1": t1_line,
        "visual_description_pair": visual_pair,
        "spectral_change_narrative": spectral,
        "analyst_description": analyst,
        "condition_c_user_prompt": user_prompt,
        "explanation": analyst,
        "inference_summary": conclusion,
    }


def load_csv_by_patch(root: Path) -> dict[str, dict]:
    csv_path = root / "data" / "raw_metadata" / "selected_patches.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "patch_id" not in df.columns:
        return {}
    return {str(r.patch_id): r.to_dict() for _, r in df.iterrows()}


def enrich_row(row: dict[str, Any], root: Path, csv_map: Optional[dict[str, dict]] = None) -> dict[str, Any]:
    csv_map = csv_map or load_csv_by_patch(root)
    analysis = analyze_patch_files(row, root, csv_map.get(row["id"]))
    texts = synthesize_narrative(row, analysis)
    out = dict(row)
    out["file_analysis"] = analysis
    out["text"] = {k: texts[k] for k in (
        "visual_description_t0",
        "visual_description_t1",
        "visual_description_pair",
        "spectral_change_narrative",
        "analyst_description",
        "condition_c_user_prompt",
        "inference_summary",
    )}
    out["explanation"] = texts["explanation"]
    out["explanation_source"] = "raster_analysis_v2"
    out["rendering_note"] = (
        "Terrain RGB stats from *_T0_rgb.tif / *_T1_rgb.tif with fixed S2 reflectance window (vmax=0.22). "
        "Spectral/SAR/change stats from aligned 512×512 .npy tensors referenced in dataset.jsonl."
    )
    return out
