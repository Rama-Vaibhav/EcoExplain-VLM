#!/usr/bin/env python3
"""Behaviour tests for the directional-claim checker (src/explain_metrics.py), sir's Major 4.4.

These tests document what the checker actually does, including its known failure modes, so the
limitations stated in report v6 (Section 2.9) are demonstrated rather than asserted. A test named
"limitation_*" passes when the checker shows the limitation.

    .venv/bin/python scripts/review/test_explain_checker.py
Writes results/review/checker_tests.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json  # noqa: E402
from src.explain_metrics import extract_claims, score_justification  # noqa: E402

# A synthetic card: NDVI fell clearly, EVI barely moved, rainfall fell.
RECORD = {"file_analysis": {
    "index_delta": {"NDVI": -0.05, "EVI": 0.005, "NDMI": -0.03, "NBR": -0.03},
    "terrain_rgb_t0": {"green_dominant_fraction": 0.9}, "terrain_rgb_t1": {"green_dominant_fraction": 0.9},
    "sar_vv_delta": -0.5, "sar_vh_delta": -0.5, "rainfall_delta_mm": -40.0,
    "soil_moisture_anomaly_t0": 0.02, "soil_moisture_anomaly_t1": -0.04,
    "temp_anomaly_t0": -1.5, "temp_anomaly_t1": 0.9,
}}


def claims(text: str) -> list[tuple[str, str]]:
    return [(c["indicator"], c["direction"]) for c in extract_claims(text)]


def consistency(text: str, pred: str = "decline") -> tuple[int, int]:
    s = score_justification(text, RECORD, pred)
    return s["n_correct"], s["n_claims"]


CASES = [
    ("correct_simple", lambda: consistency("NDVI decreased from 0.70 to 0.65.") == (1, 1)),
    ("wrong_direction_detected", lambda: consistency("NDVI increased strongly.") == (0, 1)),
    ("small_change_scored_stable", lambda: consistency("EVI increased slightly.") == (0, 1)),
    ("copied_rule_is_consistent_by_construction",
     lambda: consistency("NDVI Δ ≤ −0.02 → decline") == (1, 1)),
    ("limitation_negation_read_as_assertion",
     lambda: claims("NDVI did not decrease.") == [("NDVI", "decrease")]),
    ("limitation_hedge_scored_as_firm", lambda: consistency("NDVI may have declined a little.") == (1, 1)),
    ("limitation_compound_scored_per_indicator",
     lambda: claims("NDVI and EVI both declined.") == [("NDVI", "decrease"), ("EVI", "decrease")]),
    ("limitation_repeated_mention_counts_twice",
     lambda: consistency("NDVI decreased. As noted, NDVI decreased.") == (2, 2)),
    ("limitation_mention_without_direction_unscored",
     lambda: consistency("The NDVI values show a small change within the allowed range.") == (0, 0)),
    ("limitation_causal_claim_not_scored_as_cause",
     lambda: claims("Lower rainfall caused the NDVI decline.") == [("NDVI", "decrease"), ("rainfall", "decrease")]),
]


def main() -> None:
    results = {}
    for name, test in CASES:
        try:
            ok = bool(test())
        except Exception as exc:  # report, do not hide
            ok = f"error: {exc}"
        results[name] = ok
        print(f"{'PASS' if ok is True else 'FAIL'}  {name}")
    write_json("checker_tests.json", results)
    if not all(v is True for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
