"""Compact numeric evidence for explain-mode prompts. No class labels."""

from __future__ import annotations

from typing import Any, Optional


def _f(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _signed(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def measured_deltas(record: dict[str, Any]) -> dict[str, Optional[float]]:
    """Signed T1−T0 values used both in the prompt table and in faithfulness checks."""
    fa = record.get("file_analysis") or {}
    delta = fa.get("index_delta") or {}
    rgb0 = fa.get("terrain_rgb_t0") or {}
    rgb1 = fa.get("terrain_rgb_t1") or {}
    g0, g1 = rgb0.get("green_dominant_fraction"), rgb1.get("green_dominant_fraction")
    soil0, soil1 = fa.get("soil_moisture_anomaly_t0"), fa.get("soil_moisture_anomaly_t1")
    temp0, temp1 = fa.get("temp_anomaly_t0"), fa.get("temp_anomaly_t1")

    def sub(a: Any, b: Any) -> Optional[float]:
        try:
            return float(b) - float(a)
        except (TypeError, ValueError):
            return None

    def num(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "NDVI": num(delta.get("NDVI")),
        "EVI": num(delta.get("EVI")),
        "NDMI": num(delta.get("NDMI")),
        "NBR": num(delta.get("NBR")),
        "green_cover": sub(g0, g1),
        "VV": num(fa.get("sar_vv_delta")),
        "VH": num(fa.get("sar_vh_delta")),
        "rainfall": num(fa.get("rainfall_delta_mm")),
        "soil_moisture": sub(soil0, soil1),
        "temperature": sub(temp0, temp1),
    }


def evidence_bullets(record: dict[str, Any]) -> list[str]:
    """Verbatim facts for structured explain prompts. No class labels."""
    fa = record.get("file_analysis") or {}
    i0 = fa.get("indices_t0") or {}
    i1 = fa.get("indices_t1") or {}
    delta = fa.get("index_delta") or {}
    bullets: list[str] = []
    if i0.get("NDVI") is not None and i1.get("NDVI") is not None:
        bullets.append(
            f"Patch mean NDVI was {_f(i0.get('NDVI'))} in 2022 and {_f(i1.get('NDVI'))} in 2023 "
            f"(Δ {_signed(delta.get('NDVI'))})."
        )
    if delta.get("EVI") is not None:
        bullets.append(f"EVI change (Δ) was {_signed(delta.get('EVI'))}.")
    if delta.get("NDMI") is not None:
        bullets.append(f"NDMI change (Δ) was {_signed(delta.get('NDMI'))}.")
    if fa.get("rainfall_mm_t0") is not None and fa.get("rainfall_mm_t1") is not None:
        bullets.append(
            f"Jan–Feb rainfall was {_f(fa.get('rainfall_mm_t0'), 1)} mm in 2022 and "
            f"{_f(fa.get('rainfall_mm_t1'), 1)} mm in 2023."
        )
    if fa.get("temp_anomaly_t0") is not None and fa.get("temp_anomaly_t1") is not None:
        bullets.append(
            f"Temperature anomaly was {_signed(fa.get('temp_anomaly_t0'), 1)} °C in 2022 and "
            f"{_signed(fa.get('temp_anomaly_t1'), 1)} °C in 2023."
        )
    if fa.get("sar_vv_delta") is not None:
        bullets.append(f"SAR VV backscatter change was {_signed(fa.get('sar_vv_delta'), 2)} dB.")
    return bullets


def ndvi_class_hint(record: dict[str, Any]) -> str:
    """Operational rule text (same thresholds as closed-set probe), not the dataset gold field."""
    d = measured_deltas(record).get("NDVI")
    if d is None:
        return "Use NDVI trend only; if unclear, answer stable."
    if d <= -0.02:
        return "NDVI Δ ≤ −0.02 → decline."
    if d >= 0.02:
        return "NDVI Δ ≥ +0.02 → increase."
    return "NDVI Δ between −0.02 and +0.02 → stable."


def compact_evidence_structured(record: dict[str, Any]) -> str:
    """Compact table plus numbered ALLOWED_FACTS for cite-only explain prompts."""
    table = compact_evidence(record)
    bullets = evidence_bullets(record)
    lines = [table, "", "ALLOWED_FACTS (copy verbatim into supporting_evidence; do not invent other numbers):"]
    for i, b in enumerate(bullets, 1):
        lines.append(f"{i}. {b}")
    lines.append(f"CLASS_RULE: {ndvi_class_hint(record)}")
    return "\n".join(lines)


def compact_evidence(record: dict[str, Any]) -> str:
    """Short numeric table. Interpretive tags and candidate_class are not included."""
    fa = record.get("file_analysis") or {}
    i0 = fa.get("indices_t0") or {}
    i1 = fa.get("indices_t1") or {}
    delta = fa.get("index_delta") or {}
    rast = fa.get("delta_ndvi_raster") or {}
    frac = rast.get("fraction_below_-0.05")
    try:
        frac_txt = f"{100 * float(frac):.1f}% of pixels dropped by more than 0.05"
    except (TypeError, ValueError):
        frac_txt = "pixel-drop fraction n/a"
    rgb0 = fa.get("terrain_rgb_t0") or {}
    rgb1 = fa.get("terrain_rgb_t1") or {}
    return "\n".join(
        [
            "Evidence (T0 2022 -> T1 2023, patch means):",
            (
                f"NDVI {_f(i0.get('NDVI'))} -> {_f(i1.get('NDVI'))} "
                f"(Δ {_signed(delta.get('NDVI'))}); {frac_txt}"
            ),
            (
                f"EVI Δ {_signed(delta.get('EVI'))} | "
                f"NDMI Δ {_signed(delta.get('NDMI'))} | "
                f"NBR Δ {_signed(delta.get('NBR'))}"
            ),
            (
                "Green cover in RGB "
                f"{_f(rgb0.get('green_dominant_fraction'), 2)} -> "
                f"{_f(rgb1.get('green_dominant_fraction'), 2)}"
            ),
            (
                f"SAR VV Δ {_signed(fa.get('sar_vv_delta'), 2)} dB, "
                f"VH Δ {_signed(fa.get('sar_vh_delta'), 2)} dB"
            ),
            (
                "Rainfall Jan–Feb "
                f"{_f(fa.get('rainfall_mm_t0'), 1)} -> {_f(fa.get('rainfall_mm_t1'), 1)} mm"
            ),
            (
                "Soil moisture anomaly "
                f"{_signed(fa.get('soil_moisture_anomaly_t0'), 3)} -> "
                f"{_signed(fa.get('soil_moisture_anomaly_t1'), 3)} | "
                "Temperature anomaly "
                f"{_signed(fa.get('temp_anomaly_t0'), 1)} -> {_signed(fa.get('temp_anomaly_t1'), 1)} °C"
            ),
            f"Elevation {_f(fa.get('elevation_m'), 0)} m, slope {_f(fa.get('slope_deg'), 0)}°",
        ]
    )
