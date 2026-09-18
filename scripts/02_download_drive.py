#!/usr/bin/env python3
"""Download GeoTIFF exports from Google Drive into data/raw_tiffs/.

Three ways to get files locally:

  A) Automated (recommended after Drive export):
     python scripts/02_download_drive.py
     Requires credentials.json + one-time OAuth (token.json).

  B) Manual from drive.google.com:
     1. Open https://drive.google.com
     2. Open folder ecoexplain_vlm_raw_tiffs
     3. Select all .tif files → Download
     4. Unzip into data/drive_inbox/
     5. python scripts/02_download_drive.py --from-inbox

  C) rclone (if you use it):
     rclone copy gdrive:ecoexplain_vlm_raw_tiffs data/raw_tiffs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.drive_utils import download_drive_folder, sync_from_drive_inbox, verify_downloaded  # noqa: E402
from src.paths import DRIVE_INBOX, RAW_TIFFS, SELECTED_PATCHES_CSV, ensure_data_dirs  # noqa: E402
from src.validate_dataset import load_patch_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / organize Drive GeoTIFFs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_kanha.yaml")
    parser.add_argument("--from-inbox", action="store_true", help="Organize data/drive_inbox/*.tif only")
    parser.add_argument("--folder", type=str, default=None, help="Drive folder name override")
    args = parser.parse_args()

    ensure_data_dirs()
    cfg = yaml.safe_load(args.config.read_text())
    folder = args.folder or cfg["export"]["drive_folder"]
    patch_ids = load_patch_ids(SELECTED_PATCHES_CSV, cfg["patch"].get("pilot_target"))

    if args.from_inbox:
        result = sync_from_drive_inbox(DRIVE_INBOX, RAW_TIFFS)
        print(f"Organized {result['organized']} files from {DRIVE_INBOX}")
    else:
        print(f"Downloading from Google Drive folder: {folder}")
        print("First run opens a browser for OAuth consent.")
        files = download_drive_folder(folder, RAW_TIFFS)
        print(f"Downloaded / verified {len(files)} GeoTIFF files in {RAW_TIFFS}")

    check = verify_downloaded(RAW_TIFFS, patch_ids)
    if check["complete"]:
        print(f"SUCCESS: all {len(patch_ids)} patches have 15 GeoTIFFs each.")
    else:
        print(f"INCOMPLETE: {check['missing_count']} files still missing.")
        for name in check["missing_files"][:10]:
            print("  missing:", name)
        if check["missing_count"] > 10:
            print(f"  ... and {check['missing_count'] - 10} more")


if __name__ == "__main__":
    main()
