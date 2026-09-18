#!/usr/bin/env python3
"""End-to-end EcoExplain-VLM dataset pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)] + (extra or [])
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full dataset extraction pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_kanha.yaml")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--from-inbox", action="store_true", help="Use manual Drive download inbox")
    args = parser.parse_args()

    cfg_flag = ["--config", str(args.config)]

    if not args.skip_export:
        run("01_export_gee.py", cfg_flag)

    if not args.skip_download:
        dl_args = cfg_flag + (["--from-inbox"] if args.from_inbox else [])
        run("02_download_drive.py", dl_args)

    run("03_build_dataset.py", cfg_flag)
    run("04_validate.py")


if __name__ == "__main__":
    main()
