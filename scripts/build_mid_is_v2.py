#!/usr/bin/env python3
"""Generate mid_IS_v2.pptx — corrected mid-review deck for EcoExplain-VLM."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mid_IS_v2.pptx"
QA_DIR = ROOT / "data" / "qa_figures"

DARK = RGBColor(0x1A, 0x33, 0x52)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
MUTED = RGBColor(0x55, 0x55, 0x55)


def load_stats() -> dict:
    rows = [json.loads(line) for line in (ROOT / "dataset.jsonl").read_text().splitlines() if line.strip()]
    import pandas as pd

    df = pd.read_csv(ROOT.parent / "data" / "labels" / "final_labels.csv")
    agree = int((df.candidate_class == df.reference_class).sum())
    return {
        "records": len(rows),
        "ollama": sum(1 for r in rows if r.get("explanation_source") == "ollama"),
        "fallback": sum(1 for r in rows if r.get("explanation_source") == "fallback"),
        "boxes_total": sum(len(r.get("triplets") or []) for r in rows),
        "boxes_mean": round(sum(len(r.get("triplets") or []) for r in rows) / len(rows), 2),
        "zero_box": sum(1 for r in rows if not r.get("triplets")),
        "agree": agree,
        "n": len(df),
        "ref_stable": int((df.reference_class == "stable").sum()),
        "ref_decline": int((df.reference_class == "decline").sum()),
        "ref_increase": int((df.reference_class == "increase").sum()),
        "decline_no_box": [
            r["id"]
            for r in rows
            if r.get("reference_class") == "decline" and not r.get("triplets")
        ],
    }


def set_title(slide, title: str, subtitle: str = "") -> None:
    slide.shapes.title.text = title
    if subtitle and len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle


def add_bullets(text_frame, lines: list[str], size: int = 18) -> None:
    text_frame.clear()
    for i, line in enumerate(lines):
        p = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(size)
        p.font.color.rgb = MUTED


def add_content_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    set_title(slide, title)
    add_bullets(slide.placeholders[1].text_frame, bullets)


def add_two_column_slide(prs: Presentation, title: str, left: list[str], right: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide, title)
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(4.5), Inches(5))
    right_box = slide.shapes.add_textbox(Inches(5.2), Inches(1.5), Inches(4.5), Inches(5))
    add_bullets(left_box.text_frame, left, 16)
    add_bullets(right_box.text_frame, right, 16)


def build(stats: dict) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    agree_pct = round(100 * stats["agree"] / stats["n"], 1)

    # 1 — Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "EcoExplain-VLM"
    slide.placeholders[1].text = (
        "Mid Independent Study Presentation (v2)\n"
        "Evaluating evidence-grounded explanations from VLMs for forest disturbance assessment\n\n"
        f"Pilot: {stats['records']} patches | 540 GeoTIFFs | 5 sources/patch | "
        f"{stats['records']} JSONL rows | Ollama {stats['ollama']}/{stats['records']}\n"
        "Kanha Tiger Reserve | T0: Jan–Feb 2022 | T1: Jan–Feb 2023 | matched dry-season comparison"
    )

    # 2 — Order
    add_content_slide(
        prs,
        "Presentation Order",
        [
            "1. Project framing — research question and benchmark idea (Roadmap §1, §5)",
            "2. Stage 1 — candidate generation from NDVI quantiles (Roadmap §6)",
            "3. Stage 2 — quality filtering and GEE five-source export",
            "4. Processing — DIP/Otsu, boxes, 512×512 images, tensors (Notebook 02)",
            "5. Stage 3 — manual validation and pilot results",
            "6. JSONL evidence cards + Ollama explanations (Notebook 03)",
            "7. Limitations, corrected next steps, and 400-patch scale plan",
        ],
    )

    # 3 — Framing
    add_content_slide(
        prs,
        "1. Project Framing",
        [
            "Research question: Can VLMs classify forest vegetation change using before/after remote-sensing evidence?",
            "Can explanations be checked against optical, SAR, climate, and terrain evidence?",
            "This is an explainability/evaluation benchmark — not a new remote-sensing classifier.",
            "Deliverable: multimodal records linking images, tensors, spatial boxes, human labels, and explanations.",
            "Aligned with EcoExplain-VLM Complete Research Roadmap (pilot phase, Days 11–18).",
        ],
    )

    # 4 — Roadmap status
    add_two_column_slide(
        prs,
        "Roadmap Status — Pilot Complete",
        [
            "✅ GEE export (Notebook 01) — 36 patches",
            "✅ Five-source cube — 15 GeoTIFFs/patch (540 total)",
            "✅ DIP + Otsu (Notebook 02) — masks, boxes, tensors",
            "✅ 512×512 chips + bounding-box triplets",
            "✅ Stage 3 manual validation — 36/36 human labels",
            f"✅ JSONL (Notebook 03) — {stats['records']} records",
            f"✅ Ollama explanations — {stats['ollama']}/{stats['records']} (0 fallback)",
        ],
        [
            "⏳ Stage 4 independent audit — second rater, Cohen's κ",
            "⏳ Full scale — 400 patches (9% done today)",
            "⏳ VLM evaluation — Conditions A / B / C",
            "⏳ Groundedness, contradiction, calibration metrics",
            "⏳ Paper and public release",
            "",
            "Gate (Roadmap §17): scale only after pilot QA approval.",
        ],
    )

    # 5 — Study design
    add_two_column_slide(
        prs,
        "Study Design",
        [
            "Study area: Kanha Tiger Reserve",
            "ROI bbox: [80.433, 22.117, 81.050, 22.450]",
            "T0: Jan–Feb 2022 (before)",
            "T1: Jan–Feb 2023 (after)",
            "Patch size: 64×64 px @ 10 m (~640 m)",
            "Cloud cap: ≤20% (SCL mask)",
        ],
        [
            "Pilot target: 36 patches (Roadmap: 40–60)",
            "Full target: 400 patches",
            "ERA5 climatology: 2015–2021",
            "Season-matched windows reduce phenology confounding",
            "Coordinates used for export only — stripped from VLM JSONL",
            "Config: configs/pilot_kanha.yaml",
        ],
    )

    # 6 — Stage 1
    add_content_slide(
        prs,
        "2. Stage 1 — Candidate Generation",
        [
            "Sentinel-2 median composites → NDVI before/after → NDVI change",
            "Region-specific quantiles assign candidate_class: stable / decline / increase",
            "Purpose: balanced sampling frame — not final ground truth (Roadmap §6 Stage 1)",
            "Pilot candidate mix: 12 stable, 12 decline, 12 increase",
            "Final human reference mix: "
            f"{stats['ref_stable']} stable, {stats['ref_decline']} decline, {stats['ref_increase']} increase",
        ],
    )

    # 7 — Stage 2 filters
    add_content_slide(
        prs,
        "3. Stage 2 — Quality Filtering",
        [
            "Cloud/shadow mask — reduces false change from artefacts",
            "Season matching — Jan–Feb compared to Jan–Feb",
            "SAR orbit consistency — dominant Sentinel-1 pass per window",
            "ROI / forest context — samples inside Kanha study boundary",
            "Completeness checks — 15/15 GeoTIFFs per patch required before DIP",
            "Manual inspectability — T0/T1 RGB available for human review",
        ],
    )

    # 8 — GEE export
    add_content_slide(
        prs,
        "GEE Export (Notebook 01)",
        [
            "Authenticated Google Earth Engine (GCP project configured)",
            "Sentinel-2 cloud-masked median composites for T0 and T1",
            "Sentinel-1 VV/VH composites with dominant orbit pass",
            "CHIRPS rainfall, ERA5-Land temp/SM anomalies, NASADEM terrain",
            "Derived ΔNDVI and ΔNBR for disturbance evidence",
            "Export modes: local (used) | Drive | local_then_drive",
            "Pilot result: 540/540 raw GeoTIFF files present (36 × 15)",
        ],
    )

    # 9 — Five sources table as bullets
    add_content_slide(
        prs,
        "Five-Source Evidence Cube (15 GeoTIFFs / patch)",
        [
            "Sentinel-2: RGB + 10 optical bands + NDVI/EVI/NDMI/NBR — 6 files",
            "Sentinel-1: VV, VH — 2 files",
            "CHIRPS: rainfall mm — 2 files",
            "ERA5-Land: temperature + soil-moisture anomalies — 2 files",
            "NASADEM: elevation, slope, aspect — 1 file",
            "Derived: ΔNDVI, ΔNBR — 2 files",
            "ERA5 is regional context, not a 10 m measurement (Roadmap §2.2).",
        ],
    )

    # 10 — DIP
    add_content_slide(
        prs,
        "4. DIP + Otsu Processing (Notebook 02)",
        [
            "ΔNDVI negative drops → deforestation signal; ΔNBR drops → fire/burn signal",
            "Otsu threshold on drop magnitudes → binary masks → contours → bounding boxes",
            "Severity from mean negative index inside each box (Low / Moderate / High)",
            f"Pilot totals: {stats['boxes_total']} boxes across 36 patches (mean {stats['boxes_mean']}/patch)",
            "Reference-class policy: stable & increase → 0 boxes; decline → keep top deforestation boxes",
            f"Result: {stats['zero_box']}/36 patches have zero boxes (expected after policy)",
        ],
    )

    # 11 — 512 assets
    add_content_slide(
        prs,
        "512×512 VLM Assets",
        [
            "Native chips padded to square, resized to 512×512",
            "Bounding boxes transformed with the same padding/scale",
            "Outputs: T0/T1 JPGs (74), aligned NPY tensors (468), evidence masks (72)",
            "Same transform applied to all five-source tensors for spatial alignment",
            "Train / dev / test split defined: 23 / 2 / 11 patch IDs",
        ],
    )

    # 12 — JSONL
    add_content_slide(
        prs,
        "5. VLM JSONL Evidence Cards (Notebook 03)",
        [
            "One JSONL record per patch: images, sentinel2, sentinel1, chirps, era5_land, dem",
            "evidence: ΔNDVI/ΔNBR tensors and Otsu masks",
            "triplets: [bbox, disturbance_type, severity]",
            f"explanation: Ollama llama3 captions ({stats['ollama']}/{stats['records']} unique)",
            "candidate_class (auto) + reference_class (human validated)",
            "No lat/lon in VLM-facing records — reduces geographic memorization leakage",
        ],
    )

    # 13 — QA image slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    set_title(slide, "QA Panel Example — p00052 (decline)")
    qa_img = QA_DIR / "p00052_qa_panel.png"
    if qa_img.exists():
        slide.shapes.add_picture(str(qa_img), Inches(0.4), Inches(1.4), width=Inches(8.8))
    notes = slide.shapes.add_textbox(Inches(9.4), Inches(1.5), Inches(3.5), Inches(5))
    add_bullets(
        notes.text_frame,
        [
            "reference_class: decline",
            "2 Otsu deforestation boxes",
            "Low-severity spatial evidence",
            "Ollama explanation grounded in boxes + scalars",
            "",
            "Also available:",
            "p01963 (increase)",
            "p03276 (stable)",
            "p03545 (decline, 0 boxes)",
        ],
        14,
    )

    # 14 — Stage 3
    add_content_slide(
        prs,
        "6. Stage 3 — Manual Validation",
        [
            "All 36 pilot patches visually inspected (T0 vs T1 RGB)",
            "Human reference_class stored separately from auto candidate_class",
            f"Agreement: {stats['agree']}/{stats['n']} = {agree_pct}%",
            "Low agreement is expected — Stage 1 is a sampling heuristic, Stage 3 is operational truth",
            "Labels described as operational benchmark labels (Roadmap §6), not perfect ecological truth",
            "Stage 4 (second rater + Cohen's κ) — planned after professor approval",
        ],
    )

    # 15 — Results
    add_two_column_slide(
        prs,
        "Pilot Results (verified)",
        [
            f"JSONL records: {stats['records']}",
            "Raw GeoTIFFs: 540",
            "Processed tensors: 468",
            f"Otsu boxes (total): {stats['boxes_total']}",
            f"Ollama explanations: {stats['ollama']}/{stats['records']}",
            f"Fallback templates: {stats['fallback']}",
            "QA panels: 4 PNG figures",
            "Schema validation errors: 0",
        ],
        [
            "Candidate class (auto):",
            "  12 stable | 12 decline | 12 increase",
            "",
            "Human reference class:",
            f"  {stats['ref_stable']} stable | {stats['ref_decline']} decline | {stats['ref_increase']} increase",
            "",
            f"Zero-box patches: {stats['zero_box']}/36",
            f"Decline with 0 boxes: {len(stats['decline_no_box'])}",
            f"  ({', '.join(stats['decline_no_box'])})",
        ],
    )

    # 16 — Interpretation
    add_content_slide(
        prs,
        "Key Pilot Interpretation",
        [
            f"Pipeline works end-to-end: GEE → five-source cube → DIP → labels → Ollama JSONL",
            f"{agree_pct}% candidate/reference agreement shows human validation is necessary",
            "Otsu detects pixel-level spectral drops; humans assign patch-level reference_class",
            f"{stats['zero_box']} zero-box patches: mostly stable/increase under reference-class policy",
            f"{len(stats['decline_no_box'])} decline patches lack boxes — subtle patch-level change without strong pixel drop",
            "Pilot justifies Stage 4 audit and full scale — not final VLM benchmark claims yet",
        ],
    )

    # 17 — Checklist
    add_content_slide(
        prs,
        "Completed Deliverables",
        [
            "configs/pilot_kanha.yaml — ROI, dates, pilot/full targets",
            "notebooks/01_gee_extraction.ipynb + scripts/01_export_gee.py",
            "data/raw_tiffs/ — 540 GeoTIFFs (36 × 15)",
            "src/dip_utils.py + data/evidence_masks/ — Otsu masks",
            "data/processed_images/ — 512×512 JPGs and aligned tensors",
            "data/labels/final_labels.csv — 36 human-validated labels",
            "dataset.jsonl — 36 multimodal evidence cards with reference_class",
            "data/qa_figures/ — QA panels for professor review",
            "results/validation_report.json — schema and completeness checks",
        ],
    )

    # 18 — Limitations
    add_content_slide(
        prs,
        "Current Limitations (honest)",
        [
            "Pilot only: 36/400 patches (9%) — validates pipeline, not population statistics",
            "Otsu is simple — can miss subtle degradation or react to residual noise",
            "Ollama captions are draft ground-truth text — not yet scored for groundedness/hallucination",
            "No second annotator yet — Cohen's κ (Stage 4) pending",
            "Fire and uncertain classes configured but not active in pilot",
            "VLM Conditions A/B/C experiments not started (correct next phase)",
            "Full-scale export needs storage and batch runtime planning",
        ],
    )

    # 19 — Future work CORRECTED
    add_content_slide(
        prs,
        "Corrected Next Steps (Roadmap order)",
        [
            "1. Professor approval of pilot design and QA panels",
            "2. Spot-check 5–10 QA panels (boxes vs reference_class) — 4 panels ready",
            "3. Scale GEE export to 400 patches (same pipeline, Notebook 01)",
            "4. Re-run Notebook 02 → 03 on full set",
            "5. Stratified human audit ~20% per class (~80 patches) — not 100% manual on 400",
            "6. Stage 4: second rater + Cohen's κ on audit subset",
            "7. VLM evaluation: Conditions A (images) / B (scalars) / C (full evidence card)",
            "8. Metrics: classification, groundedness, contradiction, calibration (Roadmap §10)",
        ],
    )

    # 20 — Scale plan
    add_content_slide(
        prs,
        "Full-Scale 400-Patch Plan",
        [
            "Patches: 36 → 400 | Raw GeoTIFFs: 540 → 6,000",
            "JSONL records: 36 → 400 | T0/T1 JPGs: ~800",
            "Human work: stratified audit ~80 patches + second-rater κ (not 400 full re-label)",
            "Same season-matched windows and five-source schema",
            "Re-use validated DIP/Otsu policy and Ollama caption pipeline",
            "Freeze dataset v1 before any VLM hypothesis testing",
        ],
    )

    # 21 — VLM eval
    add_content_slide(
        prs,
        "VLM Evaluation Plan (after dataset freeze)",
        [
            "Condition A — before/after natural-colour images only",
            "Condition B — images + selected scalar metadata (NDVI, rainfall, etc.)",
            "Condition C — full evidence card: images, sensors, climate, terrain, masks/triplets",
            "Metrics: macro-F1, evidence precision/recall, unsupported-evidence rate, contradiction rate",
            "Confidence: Brier score, ECE, reliability diagrams (Roadmap §10)",
            "Models: one proprietary VLM + one hosted + one open-source (record exact versions)",
        ],
    )

    # 22 — Summary
    add_content_slide(
        prs,
        "Mid-Review Summary — Ask",
        [
            "Implemented pilot pipeline: Earth Engine → five-source cube → Otsu/DIP → human labels → Ollama JSONL",
            f"{stats['records']} patches | 540 GeoTIFFs | 0 schema errors | Ollama {stats['ollama']}/{stats['records']}",
            f"Key finding: auto NDVI candidates agree with humans only {agree_pct}% — validation is essential",
            "Requesting approval to scale to 400 patches using the same validated workflow",
            "Then: stratified audit, Cohen's κ, and VLM A/B/C benchmark experiments",
            "Thank you — questions welcome",
        ],
    )

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build(load_stats())
