#!/usr/bin/env python3
"""Pilot polish helpers: QA figures + one-page summary stats.

Run from repo root:
  python scripts/pilot_polish.py --figures
  python scripts/pilot_polish.py --summary
  python scripts/pilot_polish.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
PROC = ROOT / "data" / "processed_images"
OUT_DIR = ROOT / "data" / "qa_figures"
TRIPLETS_JSON = ROOT / "data" / "raw_metadata" / "triplets.json"
FINAL_LABELS = PARENT / "data" / "labels" / "final_labels.csv"
JSONL = ROOT / "dataset.jsonl"

# Good spread for professor slides: decline, stable, increase
DEFAULT_PATCHES = ["p00052", "p03276", "p01963"]


def load_triplets() -> dict:
    return {r["scene_id"]: r for r in json.loads(TRIPLETS_JSON.read_text())}


def export_figure(patch_id: str, rec: dict, ref_class: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = np.array(Image.open(ROOT / rec["paths"]["t0_image"]))
    t1 = np.array(Image.open(ROOT / rec["paths"]["t1_image"]))
    overlay = t1.copy()
    for box, dtype, sev in rec.get("triplets") or []:
        x0, y0, x1, y1 = box
        color = (255, 80, 40) if dtype == "Fire" else (40, 200, 80)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(overlay, f"{dtype}:{sev}", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t0)
    axes[0].set_title(f"{patch_id} T0 (2022)")
    axes[1].imshow(overlay)
    axes[1].set_title(f"{patch_id} T1 + Otsu boxes")
    axes[2].imshow(t1)
    axes[2].set_title(f"reference_class = {ref_class}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"EcoExplain-VLM pilot QA — {patch_id}", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / f"{patch_id}_qa_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def print_summary() -> None:
    final = pd.read_csv(FINAL_LABELS)
    trips = load_triplets()
    agree = (final["candidate_class"] == final["reference_class"]).mean()
    n_boxes = []
    for _, row in final.iterrows():
        rec = trips.get(row.patch_id, {})
        n_boxes.append(len(rec.get("triplets") or []))

    print("=" * 60)
    print("ECOEXPLAIN-VLM PILOT SUMMARY (copy for professor)")
    print("=" * 60)
    print(f"Study area: Kanha Tiger Reserve")
    print(f"Windows: T0 Jan–Feb 2022, T1 Jan–Feb 2023 (matched season)")
    print(f"Pilot patches: {len(final)} (human-validated 100%)")
    print(f"Reference mix: {final['reference_class'].value_counts().to_dict()}")
    print(f"Auto candidate vs human reference agreement: {agree:.0%} ({int(agree*len(final))}/{len(final)})")
    print(f"Five sources per patch: S2, S1, CHIRPS, ERA5, DEM")
    print(f"Deliverable: dataset.jsonl with 512x512 T0/T1, tensors, Otsu triplets")
    print(f"Mean Otsu boxes per patch: {np.mean(n_boxes):.1f}")
    print(f"JSONL path: {JSONL}")
    print("Note: Otsu gives pixel-level drops; human label is patch-level.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--patches", nargs="*", default=DEFAULT_PATCHES)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        args.figures = args.summary = True
    if not args.figures and not args.summary:
        parser.print_help()
        return

    final = pd.read_csv(FINAL_LABELS)
    ref_map = dict(zip(final.patch_id.astype(str), final.reference_class))
    trips = load_triplets()

    if args.summary:
        print_summary()

    if args.figures:
        for pid in args.patches:
            if pid not in trips:
                print("skip missing triplet record", pid)
                continue
            out = export_figure(pid, trips[pid], ref_map.get(pid, "?"))
            print("wrote", out)


if __name__ == "__main__":
    main()
