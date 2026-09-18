#!/usr/bin/env python3
"""Build dataset_extensive.jsonl with per-file analysis and distinct text per patch.

Reads:  dataset.jsonl (base Condition C records)
Writes: dataset_extensive.jsonl

Run from repo root:
  python scripts/build_extensive_jsonl.py
  python scripts/build_extensive_jsonl.py --replace  # also overwrite dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extensive_descriptions import enrich_row, load_csv_by_patch  # noqa: E402

IN_PATH = ROOT / "dataset.jsonl"
OUT_PATH = ROOT / "dataset_extensive.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--replace", action="store_true", help="Also write to dataset.jsonl")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing {args.input}")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    csv_map = load_csv_by_patch(ROOT)
    enriched: list[dict] = []
    errors: list[str] = []

    for row in tqdm(rows, desc="Analyze patches", unit="patch"):
        try:
            enriched.append(enrich_row(row, ROOT, csv_map))
        except Exception as exc:
            errors.append(f"{row.get('id')}: {exc}")
            enriched.append(row)

    with args.output.open("w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.replace:
        with IN_PATH.open("w", encoding="utf-8") as f:
            for row in enriched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # uniqueness check
    explanations = [r.get("explanation", "") for r in enriched]
    unique = len(set(explanations))
    print(f"Wrote {len(enriched)} records -> {args.output}")
    print(f"Distinct explanations: {unique}/{len(enriched)}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(" ", e)

    if args.replace:
        print(f"Also updated {IN_PATH}")


if __name__ == "__main__":
    main()
