#!/usr/bin/env python3
"""Professor-facing panels, PDF, HTML gallery, and summary bars.

Model order on every page: Qwen2-VL, PaliGemma, LLaVA, BLIP-2.
Patches are grouped decline, then stable, then increase.
Writes only under results/explain/.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.explain_metrics import justification_text, score_justification  # noqa: E402
from src.vlm_combos import load_jsonl, load_split_ids  # noqa: E402
from src.vlm_evidence import measured_deltas  # noqa: E402

BADGE = {"decline": "#b42318", "increase": "#1f7a3a", "stable": "#667085"}
MAJORITY = 0.34545454545454546
MODEL_ORDER = (
    "qwen_vl_custom_mlp_qwen",
    "paligemma_siglip_linear_gemma",
    "llava15_clip_mlp_vicuna",
    "blip2_clip_qformer_opt",
)
LEAKY_FILES = {
    "llava15_clip_mlp_vicuna": "vlm_combo_llava15_clip_mlp_vicuna.json",
    "paligemma_siglip_linear_gemma": "vlm_combo_paligemma_siglip_linear_gemma.json",
    "qwen_vl_custom_mlp_qwen": "vlm_combo_qwen_vl_custom_mlp_qwen.json",
    "blip2_clip_qformer_opt": "vlm_combo_blip2_clip_qformer_opt.json",
}
SHORT_NAME = {
    "llava15_clip_mlp_vicuna": "LLaVA 0.5B",
    "paligemma_siglip_linear_gemma": "PaliGemma 3B",
    "qwen_vl_custom_mlp_qwen": "Qwen2-VL 2B",
    "blip2_clip_qformer_opt": "BLIP-2 2.7B",
    "qwen_vl_finetuned_ecoexplain": "Qwen2-VL LoRA",
}


def _load_reports(explain_dir: Path) -> dict[str, dict]:
    reports = {}
    for path in explain_dir.glob("explain_*.json"):
        report = json.loads(path.read_text())
        if "predictions" not in report:
            continue
        reports[report.get("combo_id") or path.stem.replace("explain_", "")] = report
    return reports


def _by_id(report: dict | None) -> dict[str, dict]:
    if not report:
        return {}
    return {row["id"]: row for row in report.get("predictions") or []}


def _oneword_as_rows(path: Path, note: str) -> dict | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    rows = []
    for row in report.get("predictions") or []:
        pred = row.get("pred")
        rows.append(
            {
                "id": row["id"],
                "gold": row.get("gold"),
                "pred": pred,
                "correct": row.get("correct"),
                "raw_text": "",
                "parsed": {
                    "change_class": pred,
                    "justification": note,
                    "format": "oneword_only",
                },
                "source": "oneword",
            }
        )
    report = dict(report)
    report["predictions"] = rows
    report["answer_format"] = "oneword_only"
    return report


def ordered_reports(root: Path, explain_dir: Path) -> dict[str, dict]:
    """Fixed model order. PaliGemma falls back to the saved one-word test if explain mode never ran."""
    found = _load_reports(explain_dir)
    ordered = {}
    for combo_id in MODEL_ORDER:
        if combo_id in found:
            ordered[combo_id] = found[combo_id]
            continue
        if combo_id == "paligemma_siglip_linear_gemma":
            fallback = _oneword_as_rows(
                root / "results" / LEAKY_FILES[combo_id],
                "Explain mode did not run. The Hugging Face repo is gated and the saved token was rejected. "
                "This line is the earlier one-word answer, not a justification.",
            )
            if fallback:
                ordered[combo_id] = fallback
    for combo_id, report in found.items():
        if combo_id not in ordered:
            ordered[combo_id] = report
    return ordered


def _display_justification(row: dict) -> str:
    if row.get("source") == "oneword" or (row.get("parsed") or {}).get("format") == "oneword_only":
        return (row.get("parsed") or {}).get("justification") or "One-word answer only."
    parsed = row.get("parsed") or {}
    text = justification_text(parsed, row.get("raw_text") or "").strip()
    if not text or "reason: one sentence" in text.lower():
        return "No reason was produced. The model repeated the instruction."
    text = " ".join(text.split())
    if text.lower().strip(" .") in {"one sentence", "class: stable or decline or increase"}:
        return "No reason was produced. The model repeated the instruction."
    if len(text) > 420:
        text = text[:417].rsplit(" ", 1)[0] + "…"
    return text


def _faith(row: dict, record: dict) -> str:
    if row.get("source") == "oneword":
        return "n/a"
    parsed = row.get("parsed") or {}
    text = justification_text(parsed, row.get("raw_text") or "")
    if "reason: one sentence" in text.lower():
        return "n/a"
    stats = score_justification(text, record, row.get("pred"))
    if stats["faithfulness"] is None:
        return "n/a"
    return f"{stats['faithfulness']:.0%}"


def select_ids(records: dict[str, dict], reports: dict[str, dict], test_ids: list[str]) -> list[str]:
    """Three patches per gold class: decline, stable, increase."""
    chosen: list[str] = []
    for gold in ("decline", "stable", "increase"):
        buckets = {"all_right": [], "disagree": [], "all_wrong": [], "other": []}
        for pid in test_ids:
            rec = records.get(pid)
            if rec is None or rec.get("candidate_class") != gold:
                continue
            preds = []
            for report in reports.values():
                row = _by_id(report).get(pid)
                if row and row.get("pred") and row.get("source") != "oneword":
                    preds.append(row["pred"])
            if not preds:
                buckets["other"].append(pid)
                continue
            n_ok = sum(p == gold for p in preds)
            if n_ok == len(preds):
                buckets["all_right"].append(pid)
            elif len(set(preds)) > 1:
                buckets["disagree"].append(pid)
            elif n_ok == 0:
                buckets["all_wrong"].append(pid)
            else:
                buckets["other"].append(pid)
        picked = []
        while len(picked) < 3:
            progressed = False
            for name in ("all_right", "disagree", "all_wrong", "other"):
                if buckets[name]:
                    picked.append(buckets[name].pop(0))
                    progressed = True
                    if len(picked) == 3:
                        break
            if not progressed:
                break
        chosen.extend(picked)
    return chosen


def _header_numbers(rec: dict) -> str:
    d = measured_deltas(rec)
    fa = rec.get("file_analysis") or {}
    slope = fa.get("slope_deg")
    try:
        slope_txt = f"{float(slope):.1f}°"
    except (TypeError, ValueError):
        slope_txt = "n/a"

    def signed(key: str, digits: int) -> str:
        value = d.get(key)
        return "n/a" if value is None else f"{value:+.{digits}f}"

    return (
        f"ΔNDVI {signed('NDVI', 3)}    ΔNBR {signed('NBR', 3)}    "
        f"Δrain {signed('rainfall', 1)} mm    slope {slope_txt}"
    )


def render_panel(root: Path, pid: str, rec: dict, reports: dict[str, dict], out_path: Path) -> None:
    from PIL import Image

    imgs = rec.get("images") or {}
    t0 = Image.open(root / imgs["t0"]).convert("RGB")
    t1 = Image.open(root / imgs["t1"]).convert("RGB")
    npy_rel = (rec.get("evidence") or {}).get("delta_ndvi")
    delta = np.load(root / npy_rel) if npy_rel and (root / npy_rel).exists() else None

    combo_ids = [cid for cid in MODEL_ORDER if cid in reports]
    fig = plt.figure(figsize=(11.5, 12.2), dpi=300, facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.55], hspace=0.18, wspace=0.08)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    axes[0].imshow(t0)
    axes[1].imshow(t1)
    axes[0].set_title("2022 (before)", fontsize=12)
    axes[1].set_title("2023 (after)", fontsize=12)
    if delta is not None:
        lim = float(np.nanpercentile(np.abs(delta), 98)) or 0.05
        image = axes[2].imshow(delta, cmap="RdBu", vmin=-lim, vmax=lim)
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "ΔNDVI file missing", ha="center", va="center")
    axes[2].set_title("ΔNDVI", fontsize=12)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{pid}      gold: {rec.get('candidate_class')}      {_header_numbers(rec)}",
        fontsize=12,
        y=0.985,
    )

    ax_txt = fig.add_subplot(grid[1, :])
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")
    n = max(1, len(combo_ids))
    band = 1.0 / n
    for i, combo_id in enumerate(combo_ids):
        row = _by_id(reports[combo_id]).get(pid)
        top = 1.0 - i * band
        name = SHORT_NAME.get(combo_id, combo_id)
        if not row:
            ax_txt.text(
                0.01,
                top - 0.02,
                f"{i + 1}.  {name}    not scored on this patch",
                fontsize=11,
                fontweight="bold",
                color="#667085",
                va="top",
                transform=ax_txt.transAxes,
            )
            continue
        pred = row.get("pred") or "—"
        mark = "correct" if row.get("correct") else "wrong"
        color = BADGE.get(str(pred), "#344054")
        faith = _faith(row, rec)
        ax_txt.text(
            0.01,
            top - 0.02,
            f"{i + 1}.  {name}     {pred} ({mark})     faithfulness {faith}",
            fontsize=11,
            fontweight="bold",
            color=color,
            va="top",
            transform=ax_txt.transAxes,
        )
        body = textwrap.fill(_display_justification(row), width=118)
        ax_txt.text(
            0.01,
            top - 0.075,
            body,
            fontsize=9,
            color="#1d2939",
            va="top",
            linespacing=1.35,
            transform=ax_txt.transAxes,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _read_acc(path: Path) -> tuple[float | None, int | None]:
    if not path.exists():
        return None, None
    report = json.loads(path.read_text())
    return report.get("accuracy"), report.get("n_eval")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def summary_figure(root: Path, explain_dir: Path, reports: dict[str, dict]) -> Path:
    labels = []
    leaky, clean, explain = [], [], []
    for combo_id in MODEL_ORDER:
        labels.append(SHORT_NAME[combo_id])
        leaky_acc, _ = _read_acc(root / "results" / LEAKY_FILES[combo_id])
        clean_acc, _ = _read_acc(explain_dir / f"oneword_{combo_id}.json")
        report = reports.get(combo_id) or {}
        explain_acc = None if report.get("answer_format") == "oneword_only" else report.get("accuracy")
        leaky.append(leaky_acc)
        clean.append(clean_acc if clean_acc is not None else leaky_acc)
        explain.append(explain_acc)

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=160, facecolor="white")
    series = (
        ("Earlier one-word", leaky, "#98a2b3"),
        ("Clean one-word", clean, "#475467"),
        ("Explain mode", explain, "#175cd3"),
    )
    for i, (name, values, color) in enumerate(series):
        ys = [np.nan if v is None else v for v in values]
        bars = ax.bar(x + (i - 1) * width, ys, width, label=name, color=color)
        if name == "Explain mode":
            bars[0].set_label("Explain mode")
            ax.text(x[0] + (i - 1) * width, 1.02, "n=3", ha="center", va="bottom", fontsize=8, color="#175cd3")
    ax.axhline(MAJORITY, color="#b42318", linestyle="--", linewidth=1.2, label="Majority baseline 34.5%")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Test-set class accuracy, same 55 patches except where noted")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    note = (
        "Qwen explain and clean one-word were run on 3 patches, not 55.\n"
        "PaliGemma explain mode did not run (gated model). Its bars are the earlier one-word test."
    )
    fig.text(0.01, 0.01, note, fontsize=8, color="#475467")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = explain_dir / "summary_scores.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _summary_rows(root: Path, explain_dir: Path, reports: dict[str, dict], scores: dict | None) -> list[dict]:
    score_by = {row["combo_id"]: row for row in (scores or {}).get("models") or []}
    rows = []
    for combo_id in MODEL_ORDER:
        leaky_acc, leaky_n = _read_acc(root / "results" / LEAKY_FILES[combo_id])
        clean_acc, clean_n = _read_acc(explain_dir / f"oneword_{combo_id}.json")
        report = reports.get(combo_id) or {}
        explain_acc = None if report.get("answer_format") == "oneword_only" else report.get("accuracy")
        explain_n = None if report.get("answer_format") == "oneword_only" else report.get("n_eval")
        faith = (score_by.get(combo_id) or {}).get("pct_correct_claims")
        rows.append(
            {
                "name": SHORT_NAME[combo_id],
                "leaky": _pct(leaky_acc),
                "leaky_n": leaky_n,
                "clean": _pct(clean_acc),
                "clean_n": clean_n,
                "explain": _pct(explain_acc),
                "explain_n": explain_n,
                "faith": _pct(faith),
            }
        )
    return rows


def write_html(explain_dir: Path, panel_paths: list[Path], ids: list[str], records: dict, summary: Path) -> Path:
    summary_b64 = base64.b64encode(summary.read_bytes()).decode("ascii")
    blocks = [
        "<section><h2>Summary</h2>"
        f"<img alt='summary' src='data:image/png;base64,{summary_b64}'/></section>"
    ]
    current = None
    for pid, path in zip(ids, panel_paths):
        gold = (records.get(pid) or {}).get("candidate_class")
        if gold != current:
            current = gold
            blocks.append(f"<h2>Gold class: {html.escape(str(gold))}</h2>")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(
            f"<section><h3>{html.escape(pid)}</h3>"
            f"<img alt='{html.escape(pid)}' src='data:image/png;base64,{encoded}'/></section>"
        )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EcoExplain-VLM explanation report</title>
<style>
body {{ font-family: Georgia, serif; margin: 28px auto; max-width: 1100px; color: #1d2939; }}
img {{ width: 100%; height: auto; border: 1px solid #eaecf0; }}
h2 {{ margin-top: 36px; }}
</style></head><body>
<h1>EcoExplain-VLM — model justifications</h1>
<p>Kanha Tiger Reserve, January–February 2022 versus 2023. Gold label is candidate_class.
On each panel the models are always in this order: Qwen2-VL, PaliGemma, LLaVA, BLIP-2.
Decline is red, increase is green, stable is grey.</p>
<p>PaliGemma’s explain run is missing because google/paligemma-3b-mix-224 is gated and the
Hugging Face token on this machine was rejected. The panel shows its earlier one-word answer.</p>
{''.join(blocks)}
</body></html>
"""
    path = explain_dir / "explanation_report.html"
    path.write_text(page)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="", help="Comma-separated patch ids, e.g. p00052,p00010")
    args = p.parse_args()
    explain_dir = ROOT / "results" / "explain"
    reports = ordered_reports(ROOT, explain_dir)
    if not reports:
        raise SystemExit(f"No explain_*.json in {explain_dir}. Run scripts/run_explain.py first.")
    records = {r["id"]: r for r in load_jsonl(ROOT / "dataset.jsonl")}
    test_ids = load_split_ids(ROOT / "data" / "splits", "test")
    if args.ids.strip():
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        ids = select_ids(records, reports, test_ids)
    if not ids:
        raise SystemExit("No showcase patches selected.")

    panel_dir = explain_dir / "panels"
    panel_paths = []
    for pid in ids:
        out = panel_dir / f"{pid}.png"
        render_panel(ROOT, pid, records[pid], reports, out)
        panel_paths.append(out)
        print("panel", pid, records[pid].get("candidate_class"))

    scores_path = explain_dir / "explanation_scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else None
    summary = summary_figure(ROOT, explain_dir, reports)
    table_rows = _summary_rows(ROOT, explain_dir, reports, scores)

    pdf_path = explain_dir / "explanation_report.pdf"
    with PdfPages(pdf_path) as pdf:
        cover = plt.figure(figsize=(11, 8.5), dpi=150, facecolor="white")
        cover.suptitle("EcoExplain-VLM explanation report", fontsize=16, y=0.96)
        lines = ["Model order: Qwen2-VL, PaliGemma, LLaVA, BLIP-2.", ""]
        lines.append(f"{'Model':<16}{'Earlier':>10}{'Clean':>10}{'Explain':>10}{'Faithfulness':>14}")
        for row in table_rows:
            explain_cell = row["explain"] if row["explain_n"] else "not run"
            if row["explain_n"] and row["explain_n"] < 55:
                explain_cell = f"{row['explain']} (n={row['explain_n']})"
            lines.append(
                f"{row['name']:<16}{row['leaky']:>10}{row['clean'] or '—':>10}{explain_cell:>14}{row['faith']:>10}"
            )
        lines.append("")
        lines.append("Qwen explain was scored on 3 test patches. PaliGemma explain was not run.")
        cover.text(0.06, 0.78, "\n".join(lines), fontsize=11, family="monospace", va="top")
        cover_img = plt.imread(summary)
        ax = cover.add_axes((0.08, 0.06, 0.84, 0.42))
        ax.imshow(cover_img)
        ax.axis("off")
        pdf.savefig(cover)
        plt.close(cover)
        for path in panel_paths:
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11, 11.6), dpi=150)
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    html_path = write_html(explain_dir, panel_paths, ids, records, summary)
    print("wrote", pdf_path)
    print("wrote", html_path)
    print("wrote", summary)
    for row in table_rows:
        print(row["name"], "explain", row["explain"], "n", row["explain_n"])


if __name__ == "__main__":
    main()
