#!/usr/bin/env python3
"""Write spatial block train/dev/test under data/splits_spatial/ (does not overwrite random split)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.split_utils import write_spatial_block_splits  # noqa: E402


def main() -> None:
    write_spatial_block_splits(ROOT, seed=42)


if __name__ == "__main__":
    main()
