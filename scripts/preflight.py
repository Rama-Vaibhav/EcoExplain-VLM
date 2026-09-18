#!/usr/bin/env python3
"""Run unified pre-flight checks before scaling EcoExplain-VLM.

Usage:
  python scripts/preflight.py
  python scripts/preflight.py --json   # also refresh validation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import print_preflight_report, run_preflight, write_preflight_report  # noqa: E402
from src.validate_dataset import print_validation_report, run_full_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EcoExplain-VLM pre-flight checks")
    parser.add_argument("--json", action="store_true", help="Also write results/validation_report.json")
    args = parser.parse_args()

    report = run_preflight(ROOT)
    print_preflight_report(report)
    out = write_preflight_report(report)
    print(f"Report saved: {out}")

    if args.json:
        vreport = run_full_validation(ROOT)
        print_validation_report(vreport)
        vpath = ROOT / "results" / "validation_report.json"
        vpath.write_text(json.dumps(vreport, indent=2, default=str))
        print(f"Validation report: {vpath}")

    sys.exit(0 if report["ready_for_scale"] else 1)


if __name__ == "__main__":
    main()
