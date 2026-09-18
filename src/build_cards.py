"""Build roadmap §5 VLM evidence cards from processed patch records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_evidence_card(record: dict, *, explanation: str, explanation_source: str) -> dict[str, Any]:
    """Assemble a single evidence card (no lat/lon)."""
    paths = record.get("paths", {})
    scalars = record.get("scalars") or {}
    csv_stats = record.get("csv_stats") or {}
    merged = {**csv_stats, **scalars}

    def g(*keys, default=None):
        for k in keys:
            if k in merged and merged[k] is not None:
                return merged[k]
        return default

    card = {
        "id": record["scene_id"],
        "before_after": {
            "natural_colour_t0": paths.get("t0_image"),
            "natural_colour_t1": paths.get("t1_image"),
        },
        "vegetation_indices": {
            "ndvi_t0": g("ndvi_t0", "ndvi_before"),
            "ndvi_t1": g("ndvi_t1", "ndvi_after"),
            "evi_t0": g("evi_t0", "evi_before"),
            "evi_t1": g("evi_t1", "evi_after"),
            "ndmi_t0": g("ndmi_t0", "ndmi_before"),
            "ndmi_t1": g("ndmi_t1", "ndmi_after"),
            "nbr_t0": g("nbr_t0", "nbr_before"),
            "nbr_t1": g("nbr_t1", "nbr_after"),
        },
        "sar": {
            "vv_t0": g("vv_t0", "vv_before"),
            "vv_t1": g("vv_t1", "vv_after"),
            "vh_t0": g("vh_t0", "vh_before"),
            "vh_t1": g("vh_t1", "vh_after"),
            "orbit_pass": g("s1_orbit_pass"),
        },
        "climate": {
            "temperature_anomaly_t0": g("era5_temp_anomaly_t0", "era5_temp_anomaly_before"),
            "temperature_anomaly_t1": g("era5_temp_anomaly_t1", "era5_temp_anomaly_after"),
            "rainfall_mm_t0": g("chirps_rainfall_t0", "chirps_rainfall_before"),
            "rainfall_mm_t1": g("chirps_rainfall_t1", "chirps_rainfall_after"),
            "soil_moisture_anomaly_t0": g("era5_sm_anomaly_t0", "era5_sm_anomaly_before"),
            "soil_moisture_anomaly_t1": g("era5_sm_anomaly_t1", "era5_sm_anomaly_after"),
        },
        "terrain": {
            "elevation_m": g("elevation_m"),
            "slope_deg": g("slope_deg"),
            "aspect_deg": g("aspect_deg"),
        },
        "fire_indicator": g("fire_detected", default=False),
        "acquisition": {
            "season": "dry_season_jan_feb",
            "t0_window": [g("t0_start", default="2022-01-01"), g("t0_end", default="2022-02-28")],
            "t1_window": [g("t1_start", default="2023-01-01"), g("t1_end", default="2023-02-28")],
            "masked_fraction_t0": g("masked_frac_t0", "masked_frac_before"),
            "masked_fraction_t1": g("masked_frac_t1", "masked_frac_after"),
        },
        "spatial_ground_truth": {
            "triplets": [
                {"bbox": t[0], "disturbance_type": t[1], "severity": t[2]}
                for t in record.get("triplets") or []
            ]
        },
        "explanation": explanation,
        "explanation_source": explanation_source,
        "candidate_class": g("candidate_class"),
    }
    return card


def write_evidence_cards(records: list[dict], out_dir: Path, explanations: dict[str, dict]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rec in records:
        sid = rec["scene_id"]
        exp = explanations.get(sid, {})
        card = build_evidence_card(
            rec,
            explanation=exp.get("explanation", ""),
            explanation_source=exp.get("source", "unknown"),
        )
        path = out_dir / f"{sid}_evidence_card.json"
        path.write_text(json.dumps(card, indent=2, default=str))
        written.append(path)
    return written
