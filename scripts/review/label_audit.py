#!/usr/bin/env python3
"""C0.1 label and sampling audit (sir's Major 1, Major 2.1/2.4, §2.4 inconsistency).

Reconstructs the two selection batches, checks every label against the rule of its own batch,
lists the labels that the single rule printed in the paper does not reproduce, rebuilds the
dataset-construction flow, measures threshold sensitivity and the human-vs-automatic agreement.

    .venv/bin/python scripts/review/label_audit.py
Outputs: results/review/label_audit.json, results/review/off_rule_labels.csv
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, OUTCASTS, ROOT, rows, rule, split_ids, write_csv, write_json  # noqa: E402

META = ROOT / "data" / "raw_metadata"


def batch2_ids() -> list[str]:
    log = (META / "local_export_log.txt").read_text()
    seg = log[log.index("BATCH 300:372") :]
    ids: list[str] = []
    for m in re.finditer(r"\[\d+/72\] (p\d{5})", seg):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def metres(a, b) -> float:
    (x1, y1), (x2, y2) = a, b
    dx = (x2 - x1) * 111_320 * math.cos(math.radians((y1 + y2) / 2))
    return math.hypot(dx, (y2 - y1) * 111_320)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in CLASSES) / n / n
    return (po - pe) / (1 - pe)


def main() -> None:
    pool = pd.read_csv(META / "composite_stats.csv").set_index("patch_id")
    sel = pd.read_csv(META / "selected_patches.csv").set_index("patch_id")
    recs = rows()
    assert set(sel.index) == set(recs), "selected_patches.csv and dataset.jsonl disagree"

    b2_all = batch2_ids()
    b2_kept = [p for p in b2_all if p in sel.index]
    b1_all = sorted((set(sel.index) | set(OUTCASTS)) - set(b2_all))
    b1_kept = [p for p in b1_all if p in sel.index]

    q15, q85 = pool.ndvi_change.quantile(0.15), pool.ndvi_change.quantile(0.85)
    rest = pool.drop(index=b1_all)
    q15b, q85b = rest.ndvi_change.quantile(0.15), rest.ndvi_change.quantile(0.85)

    def lab(pid: str) -> str:
        return sel.loc[pid, "candidate_class"]

    b1_match = sum(rule(pool.ndvi_change[p], q15, q85) == lab(p) for p in b1_kept)
    b2_match_own = sum(rule(pool.ndvi_change[p], q15b, q85b) == lab(p) for p in b2_kept)
    off = [p for p in sel.index if rule(pool.ndvi_change[p], q15, q85) != lab(p)]

    split_of = {p: s for s in ("train", "dev", "test") for p in split_ids(s)}
    sp_of = {p: s for s in ("train", "dev", "test") for p in split_ids(s, "spatial")}
    off_rows = [
        [p, lab(p), rule(pool.ndvi_change[p], q15, q85), round(pool.ndvi_change[p], 5),
         "batch2" if p in b2_all else "batch1", split_of[p], sp_of[p]]
        for p in sorted(off)
    ]
    write_csv("off_rule_labels.csv",
              ["patch_id", "label", "stated_rule_class", "ee_dndvi", "batch", "random_split", "spatial_split"],
              off_rows)

    # Pool under the stated rule (masked-fraction branch is empty: max masked fraction <= 0.15).
    masked = (pool.masked_frac_before > 0.15) | (pool.masked_frac_after > 0.15)
    pool_cls = Counter(rule(v, q15, q85) for v in pool.ndvi_change[~masked])
    pool_bands_uncertain = int(((pool.ndvi_change > q15) & (pool.ndvi_change < q85) & (pool.ndvi_change.abs() >= 0.05)).sum())

    # Spacing actually achieved.
    pts = {p: (sel.loc[p, "lon"], sel.loc[p, "lat"]) for p in sel.index}
    within = min(metres(pts[a], pts[b]) for a, b in combinations(b1_kept, 2) if lab(a) == lab(b))
    cross = min(metres(pts[a], pts[b]) for a, b in combinations(sel.index, 2) if lab(a) != lab(b))
    close = [(a, b) for a, b in combinations(sel.index, 2) if metres(pts[a], pts[b]) < 960]

    # Class ranges on the label-making (Earth Engine) ΔNDVI.
    ranges = {c: [float(sel[sel.candidate_class == c].ndvi_change.min()),
                  float(sel[sel.candidate_class == c].ndvi_change.max())] for c in CLASSES}

    # Threshold sensitivity on the 366 (how many labels would change under other cut-offs).
    def changed(lo: float, hi: float) -> dict:
        new = {p: rule(pool.ndvi_change[p], lo, hi) for p in sel.index}
        return {"cutoffs": [round(lo, 5), round(hi, 5)],
                "labels_changed": sum(new[p] != lab(p) for p in sel.index),
                "class_counts": dict(Counter(new.values()))}
    med = pool.ndvi_change.median()
    sens = {
        "p15_p85 (stated)": changed(q15, q85),
        "p10_p90": changed(pool.ndvi_change.quantile(0.10), pool.ndvi_change.quantile(0.90)),
        "p20_p80": changed(pool.ndvi_change.quantile(0.20), pool.ndvi_change.quantile(0.80)),
        "symmetric_about_zero_+-0.02": changed(-0.02, 0.02),
        "symmetric_about_pool_median_+-0.02": changed(med - 0.02, med + 0.02),
    }

    # Human reference labels.
    ref = [(r["candidate_class"], r["reference_class"]) for r in recs.values() if r.get("reference_class")]
    rng = random.Random(42)
    boots = []
    for _ in range(5000):
        s = [rng.choice(ref) for _ in ref]
        try:
            boots.append(cohen_kappa(s))
        except ZeroDivisionError:
            pass
    boots.sort()
    human = {
        "n": len(ref),
        "agree": sum(a == b for a, b in ref),
        "cohen_kappa": cohen_kappa(ref),
        "kappa_bootstrap95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]],
        "crosstab_auto_to_human": {a: dict(Counter(b for x, b in ref if x == a)) for a in CLASSES},
        "split": dict(Counter(split_of[r["id"]] for r in recs.values() if r.get("reference_class"))),
    }

    flow = [
        ("Regular grid at 640 m spacing over the 2,350.7 km2 rectangle", 5800,
         "pilot notebook cell 12 output (.ipynb_checkpoints/Dataset_Extraction-checkpoint.ipynb); IDs run p00000-p05799"),
        ("ESA WorldCover v200: tree cover >= 0.6 and water <= 0.05 per footprint", 3668,
         "pilot notebook cell 13 output; re-checked by scripts/review/worldcover_grid_check.py"),
        ("Random sample scored in Earth Engine (3 x FULL_TARGET of 400)", len(pool),
         "data/raw_metadata/composite_stats.csv"),
        ("Masked fraction > 0.15 in either year (would be 'uncertain')", int(masked.sum()), "composite_stats.csv"),
        ("Batch 1: per-class sampling after >= 960 m spacing within class", len(b1_all), "selected_patches.csv + export log"),
        ("Batch 2: added with percentiles recomputed on the remaining pool", len(b2_all), "local_export_log.txt 'BATCH 300:372'"),
        ("Dropped: incomplete exports (batch 1 / batch 2)", f"{len(b1_all) - len(b1_kept)} / {len(b2_all) - len(b2_kept)}", "data/outcasts/"),
        ("Final benchmark", len(sel), "dataset.jsonl"),
    ]

    out = {
        "thresholds": {
            "batch1_pool_n": len(pool), "q15": q15, "q85": q85,
            "batch2_pool_n": len(rest), "q15_batch2": q15b, "q85_batch2": q85b,
        },
        "batches": {
            "batch1": {"exported": len(b1_all), "kept": len(b1_kept),
                       "classes": dict(Counter(lab(p) for p in b1_kept)), "match_own_rule": b1_match},
            "batch2": {"exported": len(b2_all), "kept": len(b2_kept),
                       "classes": dict(Counter(lab(p) for p in b2_kept)), "match_own_rule": b2_match_own},
        },
        "stated_rule_reproduces": {"all": len(sel) - len(off), "n": len(sel),
                                   "test_random": sum(p not in off for p in split_ids("test")),
                                   "test_spatial": sum(p not in off for p in split_ids("test", "spatial"))},
        "off_rule": {"n": len(off), "classes": dict(Counter(lab(p) for p in off)),
                     "random_split": dict(Counter(split_of[p] for p in off)),
                     "spatial_split": dict(Counter(sp_of[p] for p in off)),
                     "increase_with_negative_dndvi": int(((sel.candidate_class == "increase") & (sel.ndvi_change < 0)).sum())},
        "relabel_with_stated_rule": dict(Counter(rule(pool.ndvi_change[p], q15, q85) for p in sel.index)),
        "pool_classes_stated_rule": dict(pool_cls),
        "pool_uncertain_band_candidates": pool_bands_uncertain,
        "label_dndvi_ranges": ranges,
        "masked_fraction_selected": [float(sel[["masked_frac_before", "masked_frac_after"]].min().min()),
                                     float(sel[["masked_frac_before", "masked_frac_after"]].max().max())],
        "spacing_m": {"min_within_class_batch1": within, "min_cross_class": cross, "pairs_below_960": len(close),
                      "pairs_below_960_same_class": sum(lab(a) == lab(b) for a, b in close)},
        "threshold_sensitivity": sens,
        "human_reference": human,
        "flow": [{"step": s, "count": c, "source": src} for s, c, src in flow],
        "pool_mean_sd": [float(pool.ndvi_change.mean()), float(pool.ndvi_change.std())],
        "selected_mean_sd": [float(sel.ndvi_change.mean()), float(sel.ndvi_change.std())],
        "selected_abs_ge_0.05": int((sel.ndvi_change.abs() >= 0.05).sum()),
    }
    path = write_json("label_audit.json", out)
    print("wrote", path)
    for k in ("thresholds", "batches", "stated_rule_reproduces", "off_rule", "relabel_with_stated_rule",
              "pool_classes_stated_rule", "spacing_m", "masked_fraction_selected"):
        print(k, out[k])
    print("sensitivity", {k: v["labels_changed"] for k, v in sens.items()})
    print("human", {k: human[k] for k in ("n", "agree", "cohen_kappa", "kappa_bootstrap95", "split")})


if __name__ == "__main__":
    main()
