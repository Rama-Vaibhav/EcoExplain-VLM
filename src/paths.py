"""Project path helpers for ecoexplain_vlm_dataset."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_TIFFS = DATA_DIR / "raw_tiffs"
RAW_META = DATA_DIR / "raw_metadata"
PROCESSED = DATA_DIR / "processed_images"
MASKS = DATA_DIR / "evidence_masks"
LABELS = DATA_DIR / "labels"
SPLITS = DATA_DIR / "splits"
DRIVE_INBOX = DATA_DIR / "drive_inbox"
RESULTS = PROJECT_ROOT / "results"

CONFIGS = PROJECT_ROOT / "configs"
PROMPTS = PROJECT_ROOT / "prompts"
ANNOTATIONS = PROJECT_ROOT / "annotations"
NOTEBOOKS = PROJECT_ROOT / "notebooks"
SCRIPTS = PROJECT_ROOT / "scripts"

SELECTED_PATCHES_CSV = RAW_META / "selected_patches.csv"
PATCH_SCALARS_JSON = RAW_META / "patch_scalars.json"
TRIPLETS_JSON = RAW_META / "triplets.json"
EXPLANATIONS_LOG = RAW_META / "explanations_log.json"
EXPORT_MANIFEST = RAW_META / "export_manifest.json"
DATASET_JSONL = PROJECT_ROOT / "dataset.jsonl"

# Parent repo labels (source of truth for Stage 3 human validation)
PARENT_LABELS = PROJECT_ROOT.parent / "data" / "labels"
FINAL_LABELS_CSV = PARENT_LABELS / "final_labels.csv"
PARENT_SELECTED_PATCHES_CSV = PARENT_LABELS / "selected_patches.csv"


def ensure_data_dirs() -> None:
    for path in (
        RAW_TIFFS,
        RAW_META,
        PROCESSED,
        MASKS,
        LABELS,
        SPLITS,
        DRIVE_INBOX,
        RESULTS,
        ANNOTATIONS,
    ):
        path.mkdir(parents=True, exist_ok=True)
