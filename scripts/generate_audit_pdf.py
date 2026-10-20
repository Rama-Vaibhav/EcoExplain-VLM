#!/usr/bin/env python3
"""Generate the EcoExplain-VLM audit PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EcoExplain-VLM — Code, Research Plan & Implementation Audit.pdf"


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="AuditTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#1f2d3d")))
styles.add(ParagraphStyle(name="AuditSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#163a5f"), spaceBefore=16, spaceAfter=6))
styles.add(ParagraphStyle(name="AuditBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name="Mini", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=12, spaceAfter=4))


def p(text: str, style_name: str = "AuditBody") -> Paragraph:
    return Paragraph(text, styles[style_name])


def build_story():
    story = []
    story.append(p("<b>EcoExplain-VLM</b> — Code, Research Plan & Implementation Audit", "AuditTitle"))
    story.append(p("Prepared from the repository contents, generated artifacts, and validated benchmark outputs. Focus: evidence-based project archaeology, not narrative assumptions."))
    story.append(Spacer(1, 0.18 * inch))
    story.append(p("<b>Verified evidence used</b>: dataset row count, class distributions, zero-shot benchmark ranking, fine-tuning result, and code-path tracing through scripts, notebooks, and source modules."))
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Executive summary</b>", "AuditSection"))
    summary = [
        "The repository is materially real and operational: it contains the full geospatial export pipeline, DIP/Otsu processing, JSONL dataset construction, stratified split creation, multiple VLM evaluation paths, and a Qwen fine-tuning workflow.",
        "The codebase is stronger than a purely speculative prototype, but the project’s research narrative is not fully matched by the underlying evidence: the documentation and slides are more polished than the raw methodology and validation story in several places.",
        "The strongest evidence is the concrete artifact set: 366 processed evidence records, stratified train/dev/test splits, four zero-shot benchmark models, and a Qwen LoRA run with 67.3% accuracy on the test split.",
        "The main methodological caveat is that the project mixes auto-generated candidate labels (from spectral change detection) with human reference labels, and the code explicitly uses a policy that filters/keeps boxes based on reference_class when present. This is a useful engineering pipeline, but it is not a pure ground-truth benchmark by itself.",
    ]
    for item in summary:
        story.append(p(f"• {item}"))
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Project status at a glance</b>", "AuditSection"))
    table_data = [
        ["Area", "Status", "Evidence", "Assessment"],
        ["Earth Engine export pipeline", "Implemented", "scripts/01_export_gee.py; src/gee_utils.py", "Operational and connected to real ROI/date config."],
        ["DIP / Otsu disturbance extraction", "Implemented", "src/dip_utils.py", "Core logic matches the motion from NDVI/NBR masks to triplets and severity labels."],
        ["Dataset assembly", "Implemented", "scripts/03_build_dataset.py; notebooks/03_vlm_jsonl_builder.ipynb", "366 JSONL evidence cards present with multimodal metadata."],
        ["Freeze + splits", "Implemented", "scripts/05_freeze_jsonl_and_splits.py", "Stratified split writing and a manifest are present."],
        ["Zero-shot VLM evaluation", "Implemented", "src/vlm_eval.py; results/vlm_combo_comparison.json", "Four model families were benchmarked; Qwen2-VL was strongest at 56.4%."],
        ["Fine-tuning", "Implemented", "src/vlm_finetune.py; scripts/run_finetune_qwen.py", "Qwen LoRA fine-tune executed and evaluated at 67.3% on test."],
        ["Research-plan fidelity", "Mixed", "README + slides + code", "The polished plan is broadly consistent with actual code but not always a one-to-one match."],
        ["Explainability claim strength", "Partial", "src/text_utils.py; evidence cards", "Text explanations are present but best understood as narrative summaries, not a formal XAI validation protocol."],
    ]
    table = Table(table_data, colWidths=[1.3 * inch, 1.0 * inch, 1.7 * inch, 2.6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(p("<b>Verified repository facts</b>", "AuditSection"))
    bullets = [
        "Dataset row count from the actual JSONL artifact: 366 rows.",
        "Model benchmark result at zero-shot test time: Qwen2-VL best accuracy = 56.4% (n=55 test examples).",
        "Fine-tuned Qwen result: 67.3% accuracy on the same 55-example test split.",
        "The project stores candidate classes and reference classes separately: candidate_class is generated automatically, while reference_class is human-annotated when available and is intentionally used by the code policy for box filtering.",
        "The repo distinguishes extensive evidence cards from older pilot cards; freeze/split logic writes the final dataset and train/dev/test patch lists under data/splits/.",
    ]
    for item in bullets:
        story.append(p(f"• {item}"))
    story.append(Spacer(1, 0.16 * inch))

    story.append(p("<b>Planned vs implemented</b>", "AuditSection"))
    plan_table = [
        ["Research plan element", "Planned or claimed", "Actual implementation status", "Evidence"],
        ["GEE multi-source export", "Core project pillar", "Implemented", "src/gee_utils.py and scripts/01_export_gee.py"],
        ["Condition C image-to-label benchmark", "Central objective", "Implemented", "candidate_class in dataset.jsonl and evaluation scripts"],
        ["DIP-based disturbance detection", "Yes", "Implemented", "src/dip_utils.py; build_disturbance_triplets()"],
        ["Human validation / reference labels", "Yes, as quality control", "Partially implemented", "reference_class exists but only for a subset; apply_reference_class_policy() uses it when present"],
        ["Zero-shot VLM benchmarking", "Yes", "Implemented", "results/vlm_combo_comparison.json"],
        ["Fine-tuning pipeline", "Yes", "Implemented", "scripts/run_finetune_qwen.py and src/vlm_finetune.py"],
        ["Explainability / XAI validation", "Strongly framed", "Partial/engineering-level", "src/text_utils.py and evidence narrative generation, not full causal explanation tests"],
        ["Large-scale full study area rollout", "Claimed in roadmap", "Partially present as a pilot + scale-up scaffold", "configs, scripts, and data folder structure indicate a complete prototype more than a fully deployed study"],
    ]
    plan = Table(plan_table, colWidths=[1.3 * inch, 1.2 * inch, 1.55 * inch, 2.75 * inch])
    plan.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf4fb")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(plan)
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Key algorithmic and design findings</b>", "AuditSection"))
    findings = [
        "The class labels are strongly linked to spectral evidence. In src/dip_utils.py, apply_reference_class_policy() explicitly says Otsu detects spectral drops only; stable/increase patches should not carry many disturbance boxes and decline patches keep the largest boxes.",
        "The code is aware of the distinction between auto candidate_class and human reference_class. That is an important nuance: the system does not assume the model or Otsu logic is ground truth; it uses reference labels as a policy layer when available.",
        "The final benchmark uses the model’s parsed answer against candidate_class, not a fully independent annotation stream, which is a valid benchmark design for the repo but not a fully independent scientific validation of forest-change labels.",
        "The explainability layer is mostly built as a text-generation layer over evidence cards and spectral summaries. It is evidence-grounded, but it is not equivalent to an audited causal or post-hoc explainability protocol.",
    ]
    for item in findings:
        story.append(p(f"• {item}"))
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Chronology and evidence timeline</b>", "AuditSection"))
    timeline = [
        ["Stage", "Likely chronology", "What is actually present"],
        ["Design and config", "Initial ROI, dates, patch size, VLM image size", "configs/pilot_kanha.yaml"],
        ["Remote-sensing export", "Before dataset assembly", "scripts/01_export_gee.py and src/gee_utils.py"],
        ["DIP / Otsu detection", "Core processing stage", "src/dip_utils.py, extract_bounding_boxes(), build_disturbance_triplets()"],
        ["Dataset creation", "Evidence-card generation", "scripts/03_build_dataset.py and notebook 03"],
        ["Freeze + split writing", "Benchmark dataset finalization", "scripts/05_freeze_jsonl_and_splits.py"],
        ["Zero-shot evaluation", "Benchmark stage", "results/vlm_combo_comparison.json; notebooks 05–08"],
        ["Fine-tuning", "Final improvement stage", "src/vlm_finetune.py; results/vlm_combo_qwen_vl_finetuned_ecoexplain.json"],
    ]
    tl = Table(timeline, colWidths=[1.3 * inch, 1.6 * inch, 3.4 * inch])
    tl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef7e8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tl)
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Benchmark results summary</b>", "AuditSection"))
    results_table = [
        ["Rank", "Model family", "Status", "Accuracy"],
        ["1", "Qwen2-VL", "Zero-shot", "56.4%"],
        ["2", "PaliGemma", "Zero-shot", "36.4%"],
        ["3", "LLaVA-1.5", "Zero-shot", "32.7%"],
        ["4", "BLIP-2", "Zero-shot", "32.7%"],
        ["—", "Qwen2-VL fine-tuned", "LoRA fine-tune", "67.3%"],
    ]
    rt = Table(results_table, colWidths=[0.8 * inch, 1.5 * inch, 1.4 * inch, 1.2 * inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4e4d6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Conclusion</b>", "AuditSection"))
    story.append(p("The project is real, substantial, and codified. It is not merely an unimplemented plan. It contains a functioning remote-sensing data pipeline, multimodal evidence cards, benchmark code, and a successful Qwen fine-tune. However, the README and presentation materials overstate the certainty of some claims, particularly the degree to which the benchmark is a clean scientific validation rather than a structured evidence-engineering and model-comparison workflow."))
    story.append(p("The most defensible verdict is that EcoExplain-VLM is a legitimate pilot-scale research artifact: implemented, benchmarked, and partly validated, but still best described as a strong engineering + benchmarking prototype rather than a fully closed causal or XAI proof of concept."))
    story.append(Spacer(1, 0.18 * inch))

    story.append(p("<b>Primary evidence files</b>", "AuditSection"))
    evidence = [
        "README.md",
        "scripts/01_export_gee.py",
        "scripts/03_build_dataset.py",
        "scripts/05_freeze_jsonl_and_splits.py",
        "src/dip_utils.py",
        "src/gee_utils.py",
        "src/vlm_eval.py",
        "src/vlm_finetune.py",
        "results/vlm_combo_comparison.json",
        "results/vlm_combo_qwen_vl_finetuned_ecoexplain.json",
    ]
    for item in evidence:
        story.append(p(f"• {item}"))
    story.append(Spacer(1, 0.1 * inch))

    story.append(p("Prepared in the workspace using the repository’s actual artifacts and verified benchmark outputs."))
    return story


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(build_story())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
