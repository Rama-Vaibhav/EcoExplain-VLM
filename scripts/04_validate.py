#!/usr/bin/env python3
"""Validate dataset against EcoExplain-VLM roadmap requirements."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validate_dataset import print_validation_report, run_full_validation  # noqa: E402


def main() -> None:
    report = run_full_validation(ROOT)
    print_validation_report(report)
    out = ROOT / "results" / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Full report: {out}")
    sys.exit(0 if report.get("summary_ok") else 1)


if __name__ == "__main__":
    main()
