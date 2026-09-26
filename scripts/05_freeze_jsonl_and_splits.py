#!/usr/bin/env python3
"""Freeze JSONL + write stratified 366-patch train/dev/test splits.

  dataset.jsonl            -> dataset_old.jsonl   (v1 / previous)
  dataset_extensive.jsonl  -> dataset.jsonl       (raster analysis v2)

Then writes:
  data/splits/train_patches.csv
  data/splits/dev_patches.csv
  data/splits/test_patches.csv
  data/splits/split_manifest.json

Run from repo root:
  python scripts/05_freeze_jsonl_and_splits.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JSONL = ROOT / "dataset.jsonl"
EXTENSIVE = ROOT / "dataset_extensive.jsonl"
OLD = ROOT / "dataset_old.jsonl"
SPLITS = ROOT / "data" / "splits"


def freeze_jsonl(force: bool = False) -> None:
    if not EXTENSIVE.exists():
        raise SystemExit(f"Missing {EXTENSIVE}")
    if JSONL.exists() and not OLD.exists():
        shutil.copy2(JSONL, OLD)
        print(f"copied {JSONL.name} -> {OLD.name}")
    elif OLD.exists() and not force:
        print(f"{OLD.name} already exists — leaving it")
    elif force and JSONL.exists():
        shutil.copy2(JSONL, OLD)
        print(f"overwrote {OLD.name} from current {JSONL.name}")

    shutil.copy2(EXTENSIVE, JSONL)
    print(f"copied {EXTENSIVE.name} -> {JSONL.name}")

    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    src = rows[0].get("explanation_source")
    n_text = sum(1 for r in rows if (r.get("text") or {}).get("condition_c_user_prompt"))
    print(f"dataset.jsonl: {len(rows)} rows, explanation_source={src}, prompts={n_text}")


def load_patch_table() -> pd.DataFrame:
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(
        [
            {
                "patch_id": r["id"],
                "candidate_class": r.get("candidate_class") or "unknown",
                "vegetation_trend": (r.get("file_analysis") or {}).get("inference", {}).get("vegetation_trend"),
                "n_triplets": len(r.get("triplets") or []),
                "has_reference": r.get("reference_class") is not None,
            }
            for r in rows
        ]
    )
    return df


def stratified_three_way(
    df: pd.DataFrame, seed: int, train_frac: float, dev_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_idx, dev_idx, test_idx = [], [], []
    for _, grp in df.groupby("candidate_class", sort=True):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_dev = int(round(n * dev_frac))
        n_test = n - n_train - n_dev
        if n_test < 1 and n >= 3:
            n_test = 1
            n_train = max(1, n_train - 1)
        if n_dev < 1 and n >= 3:
            n_dev = 1
            n_train = max(1, n_train - 1)
        train_idx.extend(idx[:n_train])
        dev_idx.extend(idx[n_train : n_train + n_dev])
        test_idx.extend(idx[n_train + n_dev :])
    return df.loc[train_idx], df.loc[dev_idx], df.loc[test_idx]


def write_splits(seed: int, train_frac: float, dev_frac: float) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = load_patch_table()
    test_frac = 1.0 - train_frac - dev_frac
    if test_frac <= 0:
        raise SystemExit("train_frac + dev_frac must be < 1")

    train, dev, test = stratified_three_way(df, seed, train_frac, dev_frac)

    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        out = SPLITS / f"{name}_patches.csv"
        part.sort_values("patch_id").to_csv(out, index=False)
        print(f"{name:5} {len(part):3}  {dict(Counter(part.candidate_class))}")

    manifest = {
        "n_total": int(len(df)),
        "seed": seed,
        "fractions": {"train": train_frac, "dev": dev_frac, "test": round(test_frac, 4)},
        "counts": {"train": int(len(train)), "dev": int(len(dev)), "test": int(len(test))},
        "class_balance": {
            split: dict(Counter(part.candidate_class))
            for split, part in [("train", train), ("dev", dev), ("test", test)]
        },
        "note": "Stratified on candidate_class. Train/eval labels are candidate_class unless reference_class is set.",
    }
    (SPLITS / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", SPLITS / "split_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--skip-freeze", action="store_true")
    parser.add_argument("--force-old", action="store_true")
    args = parser.parse_args()

    if not args.skip_freeze:
        freeze_jsonl(force=args.force_old)
    write_splits(args.seed, args.train_frac, args.dev_frac)


if __name__ == "__main__":
    main()
