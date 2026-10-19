"""Train sampling and spatial block splits."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLASSES = ("decline", "stable", "increase")


def select_train_records(
    records: list[dict[str, Any]],
    max_n: int | None,
    *,
    balanced: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Cap training size. ``balanced`` draws evenly across candidate_class (with replacement if needed)."""
    if max_n is None or max_n <= 0:
        return records
    if not balanced:
        return records[:max_n]

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_class[str(rec.get("candidate_class") or "unknown")].append(rec)

    per = max_n // len(CLASSES)
    extra = max_n % len(CLASSES)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for i, cls in enumerate(CLASSES):
        need = per + (1 if i < extra else 0)
        pool = list(by_class.get(cls, []))
        if not pool:
            continue
        rng.shuffle(pool)
        for rec in pool:
            if len([p for p in picked if p.get("candidate_class") == cls]) >= need:
                break
            picked.append(rec)
        while len([p for p in picked if p.get("candidate_class") == cls]) < need:
            picked.append(rng.choice(pool))
    rng.shuffle(picked)
    return picked[:max_n]


def training_class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("candidate_class")) for r in records))


def write_spatial_block_splits(
    root: Path,
    *,
    out_dir: Path | None = None,
    seed: int = 42,
    n_lon_bins: int = 5,
    n_lat_bins: int = 4,
    train_frac: float = 0.70,
    dev_frac: float = 0.15,
) -> Path:
    """Assign patches to lon/lat blocks; hold out whole blocks for dev/test (reduces spatial leakage)."""
    jsonl = root / "dataset.jsonl"
    meta = root / "data" / "raw_metadata" / "selected_patches.csv"
    out = out_dir or (root / "data" / "splits_spatial")
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    df = pd.read_csv(meta)
    df = df.rename(columns={"patch_id": "patch_id"})
    ids = {r["id"] for r in rows}
    df = df[df["patch_id"].isin(ids)].copy()
    if df.empty:
        raise RuntimeError("No overlap between dataset.jsonl and selected_patches.csv")

    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    df["lon_bin"] = np.clip(
        ((df["lon"] - lon_min) / max(lon_max - lon_min, 1e-9) * n_lon_bins).astype(int),
        0,
        n_lon_bins - 1,
    )
    df["lat_bin"] = np.clip(
        ((df["lat"] - lat_min) / max(lat_max - lat_min, 1e-9) * n_lat_bins).astype(int),
        0,
        n_lat_bins - 1,
    )
    df["block_id"] = df["lon_bin"] * n_lat_bins + df["lat_bin"]

    rng = np.random.RandomState(seed)
    blocks = []
    for bid, grp in df.groupby("block_id"):
        blocks.append(
            {
                "block_id": int(bid),
                "n": len(grp),
                "classes": dict(Counter(grp["candidate_class"])),
            }
        )
    block_df = pd.DataFrame(blocks).sort_values("block_id")
    n_blocks = len(block_df)
    n_test = max(1, int(round(n_blocks * (1.0 - train_frac - dev_frac))))
    n_dev = max(1, int(round(n_blocks * dev_frac)))
    n_train = n_blocks - n_test - n_dev
    if n_train < 1:
        raise RuntimeError("Too few spatial blocks for train/dev/test; reduce bins or fractions")

    order = block_df["block_id"].to_numpy()
    rng.shuffle(order)
    test_blocks = set(order[:n_test])
    dev_blocks = set(order[n_test : n_test + n_dev])
    train_blocks = set(order[n_test + n_dev :])

    def assign_split(bid: int) -> str:
        if bid in test_blocks:
            return "test"
        if bid in dev_blocks:
            return "dev"
        return "train"

    df["split"] = df["block_id"].map(assign_split)
    parts = {name: df[df["split"] == name] for name in ("train", "dev", "test")}
    for name, part in parts.items():
        slim = part[["patch_id", "candidate_class", "lon", "lat", "block_id"]].sort_values("patch_id")
        slim.to_csv(out / f"{name}_patches.csv", index=False)
        print(f"{name:5} {len(slim):3}  {dict(Counter(slim.candidate_class))}  blocks={part.block_id.nunique()}")

    manifest = {
        "n_total": int(len(df)),
        "seed": seed,
        "scheme": "spatial_blocks",
        "n_lon_bins": n_lon_bins,
        "n_lat_bins": n_lat_bins,
        "train_blocks": [int(x) for x in sorted(train_blocks)],
        "dev_blocks": [int(x) for x in sorted(dev_blocks)],
        "test_blocks": [int(x) for x in sorted(test_blocks)],
        "fractions": {"train": train_frac, "dev": dev_frac, "test": round(1 - train_frac - dev_frac, 4)},
        "counts": {k: int(len(v)) for k, v in parts.items()},
        "class_balance": {k: dict(Counter(v["candidate_class"])) for k, v in parts.items()},
        "note": "Whole lon/lat blocks held out for dev/test. Primary paper table may still use data/splits/ (random stratified).",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out / "split_manifest.json")
    return out
