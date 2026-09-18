# EcoExplain-VLM Dataset Extraction

**Your main workspace.** This folder implements the full five-source dataset pipeline aligned with `EcoExplain-VLM_Complete_Research_Roadmap.pdf`.

## Project layout (roadmap §13)

```
ecoexplain_vlm_dataset/
├── configs/pilot_kanha.yaml      # ROI, dates, export mode
├── data/
│   ├── raw_metadata/             # patch list, scalars, triplets
│   ├── raw_tiffs/                # 15 GeoTIFFs × N patches (from GEE)
│   ├── drive_inbox/              # manual Drive downloads go here
│   ├── processed_images/         # 512×512 RGB + aligned tensors
│   ├── evidence_masks/           # Otsu masks
│   ├── labels/                   # manual validation (roadmap Stage 3)
│   └── splits/                   # train/dev/test patch IDs
├── notebooks/                    # interactive workflow (01 → 03)
├── scripts/                      # same pipeline as CLI (recommended)
├── src/                          # shared Python modules
├── prompts/                      # VLM JSON output schema
├── annotations/                  # per-patch evidence cards (no lat/lon)
├── dataset.jsonl                 # final VLM training/eval records
└── results/                      # validation reports
```

## Five data sources per patch

| Source | Variables | Files per patch |
|--------|-----------|---------------|
| Sentinel-2 | 10 bands + NDVI/EVI/NDMI/NBR + RGB | 8 |
| Sentinel-1 | VV, VH | 2 |
| CHIRPS | Rainfall (mm) | 2 |
| ERA5-Land | Temp + soil-moisture anomalies | 2 |
| NASADEM | Elevation, slope, aspect | 1 |
| Derived | ΔNDVI, ΔNBR | 2 |
| **Total** | | **15 GeoTIFFs** |

## Quick start

```bash
cd ecoexplain_vlm_dataset
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Export from Earth Engine (writes directly to data/raw_tiffs/)
python scripts/01_export_gee.py

# 2) If you used Drive export instead, download files:
python scripts/02_download_drive.py              # automated (OAuth)
# OR: download zip from drive.google.com → unzip to data/drive_inbox/
python scripts/02_download_drive.py --from-inbox

# 3) Build DIP masks, 512×512 images, dataset.jsonl
python scripts/03_build_dataset.py

# 4) Validate against roadmap
python scripts/04_validate.py

# Or run everything:
python scripts/run_pipeline.py
```

## Google Drive workflow

Earth Engine cannot push files to your laptop automatically. Two supported paths:

### Path A — Local export (default, what you already used)
`geemap.ee_export_image` writes GeoTIFFs straight into `data/raw_tiffs/`. **No Drive step needed.**

### Path B — Drive export + download
1. Set `export.mode: drive` in `configs/pilot_kanha.yaml`
2. Run `python scripts/01_export_gee.py` → tasks appear in [GEE Tasks](https://code.earthengine.google.com/tasks)
3. When tasks complete, files land in Google Drive folder `ecoexplain_vlm_raw_tiffs`
4. Download:
   - **Automated:** `python scripts/02_download_drive.py` (needs `credentials.json` from Google Cloud)
   - **Manual:** open https://drive.google.com → folder → Download → unzip into `data/drive_inbox/` → `python scripts/02_download_drive.py --from-inbox`

## Current pilot status

Run validation anytime:

```bash
python scripts/04_validate.py
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_gee_extraction.ipynb` | GEE auth, export, patch scalars |
| `02_dip_auto_classify.ipynb` | Otsu + bounding boxes + 512×512 tensors |
| `03_vlm_jsonl_builder.ipynb` | Build `dataset.jsonl` evidence cards |

Regenerate notebooks from templates:

```bash
python notebooks/_build_notebooks.py
```

## Roadmap alignment checklist

| Roadmap requirement | Status |
|---------------------|--------|
| Matched seasonal windows (Jan–Feb 2022 vs 2023) | ✅ |
| Five-source evidence cube | ✅ |
| Before/after RGB + indices + SAR + climate + DEM | ✅ |
| No lat/lon in VLM records | ✅ |
| Acquisition season + QA metadata | ✅ |
| Manual validation (Stage 3) | ⏳ use `data/labels/` |
| Independent audit / Cohen's kappa (Stage 4) | ⏳ future |
| Structured VLM output schema | ✅ `prompts/vlm_output_schema.json` |

## Google Cloud OAuth (for automated Drive download)

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs → enable **Google Drive API**
2. Credentials → Create OAuth client ID → Desktop app
3. Download JSON → save as `credentials.json` in this folder
4. Run `python scripts/02_download_drive.py` once (browser opens for consent)
5. Token cached in `token.json`

**Never commit `credentials.json` or `token.json`.**
