#!/usr/bin/env python3
"""Move incomplete patches to outcasts/ and old pilot extras to pilot_reference/.

Run from repo root:
  python scripts/organize_patch_folders.py
  python scripts/organize_patch_folders.py --drop-outcasts-from-csv
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_tiffs"
META = ROOT / "data" / "raw_metadata"
OUTCASTS = ROOT / "data" / "outcasts"
PILOT_REF = ROOT / "data" / "pilot_reference"
MANIFEST = META / "patch_folder_manifest.json"
EXPORTS_PER_PATCH = 15


def patch_ids_in_folder(folder: Path) -> set[str]:
    ids: set[str] = set()
    for p in folder.glob("*.tif"):
        m = re.match(r"^(p\d+)_", p.name)
        if m:
            ids.add(m.group(1))
    return ids


def count_files(pid: str, folder: Path) -> int:
    return len(list(folder.glob(f"{pid}_*.tif")))


def move_patch_files(pid: str, src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in src.glob(f"{pid}_*.tif"):
        shutil.move(str(f), str(dest / f.name))
        moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop-outcasts-from-csv",
        action="store_true",
        help="Remove outcast patch IDs from selected_patches.csv (366 clean rows).",
    )
    args = parser.parse_args()

    csv_path = META / "selected_patches.csv"
    selected = pd.read_csv(csv_path)
    sel_ids = set(selected["patch_id"].astype(str))

    per_main = {pid: count_files(pid, RAW) for pid in sel_ids if count_files(pid, RAW) > 0}
    outcast_ids = sorted(pid for pid, n in per_main.items() if n < EXPORTS_PER_PATCH)
    extra_ids = sorted(patch_ids_in_folder(RAW) - sel_ids)

    manifest = {
        "exports_per_patch": EXPORTS_PER_PATCH,
        "outcasts": {},
        "pilot_reference": {},
    }

    print(f"Outcasts ({len(outcast_ids)}):", outcast_ids)
    for pid in outcast_ids:
        n = move_patch_files(pid, RAW, OUTCASTS)
        manifest["outcasts"][pid] = {"files_moved": n, "reason": "incomplete_export"}
        print(f"  moved {pid}: {n} files -> {OUTCASTS}")

    print(f"Pilot reference ({len(extra_ids)}):", extra_ids)
    for pid in extra_ids:
        n = move_patch_files(pid, RAW, PILOT_REF)
        manifest["pilot_reference"][pid] = {"files_moved": n, "reason": "pre_372_pilot"}
        print(f"  moved {pid}: {n} files -> {PILOT_REF}")

    if args.drop_outcasts_from_csv and outcast_ids:
        clean = selected[~selected["patch_id"].astype(str).isin(outcast_ids)].copy()
        clean.to_csv(csv_path, index=False)
        print(f"Updated {csv_path}: {len(selected)} -> {len(clean)} rows")
        manifest["selected_patches_rows"] = len(clean)
    else:
        manifest["selected_patches_rows"] = len(selected)

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {MANIFEST}")
    remaining = patch_ids_in_folder(RAW)
    print(f"Clean patches in raw_tiffs: {len(remaining)}")


if __name__ == "__main__":
    main()
