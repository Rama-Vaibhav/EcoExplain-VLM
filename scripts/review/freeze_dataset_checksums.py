#!/usr/bin/env python3
"""C0.0: SHA-256 checksums for frozen dataset artifacts (gate G0).

Writes: results/review/dataset_checksums.json
Run from repo root: python scripts/review/freeze_dataset_checksums.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "review"
FILES = [
    ROOT / "dataset.jsonl",
    ROOT / "data" / "raw_metadata" / "selected_patches.csv",
    ROOT / "data" / "raw_metadata" / "composite_stats.csv",
    ROOT / "data" / "splits" / "train_patches.csv",
    ROOT / "data" / "splits" / "dev_patches.csv",
    ROOT / "data" / "splits" / "test_patches.csv",
    ROOT / "data" / "splits_spatial" / "train_patches.csv",
    ROOT / "data" / "splits_spatial" / "dev_patches.csv",
    ROOT / "data" / "splits_spatial" / "test_patches.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in FILES:
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(ROOT),
        "files": rows,
    }
    out_path = OUT / "dataset_checksums.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)} ({len(rows)} files)")
    for r in rows:
        print(f"  {r['sha256'][:12]}…  {r['path']}")


if __name__ == "__main__":
    main()
