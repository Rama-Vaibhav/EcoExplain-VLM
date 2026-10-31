#!/usr/bin/env python3
"""How the full-card leak works (sir's Major 7 / model-evaluation note on full vs compact).

The full card ends with "Inference for ...: vegetation_trend=<word>, ..., candidate_class=<label>".
The trend word comes from fixed ΔNDVI thresholds and disagrees with the percentile label on some
patches. If a model's full-card answers follow the trend word where it disagrees with the label,
the answer comes from the summary line, not from the measurements.

    .venv/bin/python scripts/review/leak_mechanism.py
Output: results/review/leak_mechanism.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, rows, write_json  # noqa: E402
from src.vlm_combos import encoder_text  # noqa: E402


def trend_class(word: str) -> str:
    return "decline" if "decline" in word else ("increase" if "green" in word else "stable")


def preds(rel: str) -> dict[str, str]:
    return {p["id"]: p.get("pred") for p in json.loads((RESULTS / rel).read_text())["predictions"]}


def main() -> None:
    recs = rows()
    with (ROOT / "data" / "splits" / "test_patches.csv").open() as f:
        test = list(csv.DictReader(f))
    ids = [r["patch_id"] for r in test]
    trend = {r["patch_id"]: trend_class(r["vegetation_trend"]) for r in test}
    gold = {r["patch_id"]: r["candidate_class"] for r in test}
    disagree = [i for i in ids if trend[i] != gold[i]]

    # Where does the trend word occur in each input?
    where = Counter()
    for i in ids:
        word = recs[i]["file_analysis"]["inference"]["vegetation_trend"]
        for mode in ("full", "evidence_only", "compact"):
            text = encoder_text(recs[i], text_mode=mode)
            where[(mode, f"vegetation_trend={word}" in text)] += 1

    runs = {
        "qwen_full": "vlm_combo_qwen_vl_custom_mlp_qwen.json",
        "llava_full": "vlm_combo_llava15_clip_mlp_vicuna.json",
        "qwen_compact": "protocol/random_compact/vlm_combo_qwen_vl_custom_mlp_qwen.json",
    }
    evid = RESULTS / "protocol/random_evidence_only/vlm_combo_qwen_vl_custom_mlp_qwen_336.json"
    if evid.exists():
        runs["qwen_evidence_only_336"] = str(evid.relative_to(RESULTS))
    out = {
        "test_n": len(ids),
        "trend_vs_label_disagree": len(disagree),
        "disagree_pairs_label_trend": {f"{a}|{b}": n for (a, b), n in Counter((gold[i], trend[i]) for i in disagree).items()},
        "summary_line_present": {f"{m}:{flag}": n for (m, flag), n in where.items()},
        "runs": {},
    }
    for name, rel in runs.items():
        p = preds(rel)
        out["runs"][name] = {
            "matches_trend": sum(p[i] == trend[i] for i in ids),
            "matches_label": sum(p[i] == gold[i] for i in ids),
            "on_disagreeing_follows_trend": sum(p[i] == trend[i] for i in disagree),
            "on_disagreeing_follows_label": sum(p[i] == gold[i] for i in disagree),
            "answers": dict(Counter(p[i] for i in ids)),
        }
    print("wrote", write_json("leak_mechanism.json", out))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
