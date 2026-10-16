#!/usr/bin/env python3
"""Count label leaks in evidence_only and compact-evidence prompts. Must print 0."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlm_combos import encoder_text, load_jsonl, text_has_label_leak  # noqa: E402
from src.vlm_evidence import compact_evidence  # noqa: E402


def main() -> None:
    records = load_jsonl(ROOT / "dataset.jsonl")
    leaks = 0
    for rec in records:
        gold = rec.get("candidate_class")
        for text in (
            encoder_text(rec, text_mode="evidence_only"),
            encoder_text(rec, text_mode="minimal"),
            compact_evidence(rec),
        ):
            if text_has_label_leak(text, gold):
                leaks += 1
                print("LEAK", rec.get("id"), text_has_label_leak(text, gold))
    print(f"records={len(records)} leak_inputs={leaks}")
    if leaks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
