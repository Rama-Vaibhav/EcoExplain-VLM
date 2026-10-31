#!/usr/bin/env python3
"""C0.9 run manifest and evidence-card schema (sir's Major 7.1, 7.3-7.5 and 8.4-8.7).

One row per reported run with the checkpoint, images, text, image size, decoding, precision,
parser and answered n, read from the result JSONs (with the code defaults where an older file did
not record a field), plus the field list of the compact evidence table.

    .venv/bin/python scripts/review/run_manifest.py
Outputs: results/review/run_manifest.json, results/review/run_manifest.md
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, RESULTS, REVIEW, ROOT, write_json  # noqa: E402
from stats_report import RUNS  # noqa: E402

IMAGES = {
    "pali_ndvi": "T0 and T1 chips, resized by the PaliGemma processor to 224 px",
    "blip_img": "T0|T1 stitched side by side into one image (BLIP-2 takes one image)",
    "blip_compact": "T0|T1 stitched side by side into one image",
    "sp_blip_compact": "T0|T1 stitched side by side into one image",
}
TEXT = {
    "full card incl. summary line": "full evidence card, ending with the summary line that prints the label",
    "card NDVI line only": "the card's NDVI line (label-free) + fixed one-word instruction",
    "stitched images, no card": "fixed one-word instruction only",
    "evidence-only": "full card without the summary line and without label words",
    "compact table": "label-free compact table (same text for every model)",
    "structured cite-only": "compact table + ALLOWED_FACTS + CLASS_RULE (±0.02 on card ΔNDVI) + JSON template",
    "free-form JSON explanation": "compact table + free-form JSON explanation request (no rule)",
    "full card at train and test": "full card (with summary line)",
}
CARD_SCHEMA = [
    ("NDVI T0, T1 and ΔNDVI", "Sentinel-2 L2A median composite; mean over the zero-padded 512×512 canvas", "unitless", "3 decimals", "diluted by padding"),
    ("Share of pixels with ΔNDVI < −0.05", "padded ΔNDVI raster (padding counts as 0)", "%", "1 decimal", "diluted by padding"),
    ("ΔEVI, ΔNDMI, ΔNBR", "as NDVI", "unitless", "3 decimals, signed", "diluted by padding"),
    ("Green cover in RGB, T0 and T1", "share of native RGB pixels whose green band dominates", "fraction", "2 decimals", "native pixels (not padded)"),
    ("SAR ΔVV, ΔVH", "Sentinel-1 GRD IW descending median, dB; mean over padded canvas", "dB", "2 decimals, signed", "diluted by padding"),
    ("Rainfall Jan–Feb, T0 and T1", "CHIRPS daily sum over the window; mean of a 5,566 m box at the patch centre", "mm", "1 decimal", "regional context"),
    ("Soil-moisture anomaly, T0 and T1", "ERA5-Land top layer, month-matched vs 2015–2021; 11,132 m box", "m³/m³", "3 decimals, signed", "regional context"),
    ("Temperature anomaly, T0 and T1", "ERA5-Land 2 m air temperature, month-matched vs 2015–2021; 11,132 m box", "°C", "1 decimal, signed", "regional context"),
    ("Elevation, slope", "NASADEM mean over the footprint at 30 m", "m, °", "0 decimals", "static"),
]


def git_date(rel: str) -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", f"results/{rel}"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out or "uncommitted"
    except Exception:
        return "?"


def main() -> None:
    rows = []
    for key, track, label, protocol, split, rel in RUNS:
        path = RESULTS / rel
        if not path.exists():
            rows.append({"key": key, "status": "pending", "file": f"results/{rel}"})
            continue
        d = json.loads(path.read_text())
        finetuned = "LoRA" in label
        side = d.get("max_side")
        if side is None:
            side = 384 if finetuned else (336 if d.get("laptop") else "native")
        text_key = next((k for k in TEXT if protocol.startswith(k)), None)
        preds = d.get("predictions") or []
        rows.append({
            "key": key, "track": track, "model": label, "split": split, "file": f"results/{rel}",
            "date": git_date(rel),
            "checkpoint": d.get("hf_model_id") or "Qwen/Qwen2-VL-2B-Instruct",
            "adapter": "LoRA r=8, alpha=16 (adapter weights on Ada)" if finetuned else "none",
            "images": IMAGES.get(key, "T0 and T1 chips as two images via the model's chat template"),
            "image_max_side_px": side,
            "text": TEXT.get(text_key, protocol),
            "decoding": {"greedy": not (d.get("generation") or {}).get("do_sample", False),
                         "max_new_tokens": (d.get("generation") or {}).get("max_new_tokens")},
            "precision": "fp16 (Apple MPS)" if d.get("laptop") else "fp16 base + LoRA (CUDA, Ada)",
            "parser": "JSON change_class field, else class word (src/vlm_eval.py::parse_explanation)"
                      if track == "C" else "first word / class word (src/vlm_eval.py::parse_class)",
            "status": d.get("status", "ok"),
            "n": len(preds), "answered": sum(p.get("pred") in CLASSES for p in preds),
        })
    out = {"runs": rows, "card_schema": [dict(zip(("field", "source", "unit", "rounding", "note"), r)) for r in CARD_SCHEMA],
           "software": {"mac": "transformers 4.57.6, torch 2.8.0, Python 3.9 (.venv)",
                        "ada": "peft 0.21.0 (adapter_config.json); other versions not recorded",
                        "earth_engine_collections": ["COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/S1_GRD",
                                                     "UCSB-CHG/CHIRPS/DAILY", "ECMWF/ERA5_LAND/MONTHLY_AGGR",
                                                     "NASA/NASADEM_HGT/001", "ESA/WorldCover/v200"]}}
    write_json("run_manifest.json", out)
    lines = ["| Run | Split | Checkpoint | Images (max side) | Text | Tokens | Precision | Answered | Date |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "pending":
            lines.append(f"| {r['key']} | pending | | | | | | | |")
            continue
        lines.append(f"| {r['key']} | {r['split']} | {r['checkpoint']} | {r['images']} ({r['image_max_side_px']}) | {r['text']} | "
                     f"{r['decoding']['max_new_tokens']} | {r['precision']} | {r['answered']}/{r['n']} ({r['status']}) | {r['date']} |")
    (REVIEW / "run_manifest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
