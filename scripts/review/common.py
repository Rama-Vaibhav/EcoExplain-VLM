"""Shared helpers for the Draft 1 review scripts (scripts/review/*.py).

Everything here reads frozen inputs only (dataset.jsonl, splits, raw_metadata, result JSONs);
nothing writes outside results/review/.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
REVIEW = RESULTS / "review"
CLASSES = ("decline", "stable", "increase")

# Patches dropped after export because one or more of the 15 rasters was missing (data/outcasts/).
OUTCASTS = ("p00695", "p00796", "p03073", "p03209", "p04622", "p04819")


def rows() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "dataset.jsonl").open() if l.strip()}


def split_ids(split: str, scheme: str = "random") -> list[str]:
    folder = "splits" if scheme == "random" else "splits_spatial"
    with (ROOT / "data" / folder / f"{split}_patches.csv").open() as f:
        return [r["patch_id"] for r in csv.DictReader(f)]


def card_dndvi(rec: dict) -> float:
    """ΔNDVI as printed on the evidence card (mean over the zero-padded 512×512 canvas)."""
    return float(rec["file_analysis"]["index_delta"]["NDVI"])


def rule(v: float, lo: float, hi: float) -> str:
    return "decline" if v <= lo else ("increase" if v >= hi else "stable")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(a_right_b_wrong: int, b_right_a_wrong: int) -> float:
    """Exact two-sided McNemar test (binomial on the discordant pairs, p=0.5)."""
    n = a_right_b_wrong + b_right_a_wrong
    if n == 0:
        return 1.0
    k = min(a_right_b_wrong, b_right_a_wrong)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone)."""
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    out, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[key]))
        out[key] = running
    return out


def confusion(gold: list[str], pred: list[Optional[str]]) -> dict[str, dict[str, int]]:
    """Rows = gold class, columns = predicted class (None = no parsable answer)."""
    cols = list(CLASSES) + ["none"]
    m = {g: {c: 0 for c in cols} for g in CLASSES}
    for g, p in zip(gold, pred):
        m[g][p if p in CLASSES else "none"] += 1
    return m


def per_class(gold: list[str], pred: list[Optional[str]]) -> dict[str, dict[str, float]]:
    out = {}
    for c in CLASSES:
        tp = sum(p == c and g == c for g, p in zip(gold, pred))
        fp = sum(p == c and g != c for g, p in zip(gold, pred))
        fn = sum(p != c and g == c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def summarise(gold: list[str], pred: list[Optional[str]]) -> dict:
    n = len(gold)
    k = sum(g == p for g, p in zip(gold, pred))
    lo, hi = wilson(k, n)
    pc = per_class(gold, pred)
    answered = sum(p in CLASSES for p in pred)
    return {
        "k": k,
        "n": n,
        "answered": answered,
        "accuracy": k / n,
        "ci95": [lo, hi],
        "macro_f1": sum(v["f1"] for v in pc.values()) / 3,
        "per_class": pc,
        "confusion": confusion(gold, pred),
        "answers": dict(Counter(p if p in CLASSES else "none" for p in pred)),
    }


def load_preds(rel: str, ids: Iterable[str]) -> list[Optional[str]]:
    data = json.loads((RESULTS / rel).read_text())
    by_id = {p["id"]: p.get("pred") for p in data.get("predictions") or []}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{rel}: {len(missing)} test ids missing, e.g. {missing[:3]}")
    return [by_id[i] for i in ids]


def write_json(name: str, obj) -> Path:
    REVIEW.mkdir(parents=True, exist_ok=True)
    path = REVIEW / name
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path


def write_csv(name: str, header: list[str], table: list[list]) -> Path:
    REVIEW.mkdir(parents=True, exist_ok=True)
    path = REVIEW / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    return path
