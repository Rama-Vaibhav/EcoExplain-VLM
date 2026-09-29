# EcoExplain-VLM

Benchmark and fine-tuning pipeline for **Condition C** forest-disturbance classification at **Kanha Tiger Reserve** (dry season **Jan–Feb 2022 vs 2023**).

Each patch provides **T0 + T1 RGB** and **spectral evidence text**; the model predicts **`stable` | `decline` | `increase`** (label: `candidate_class` in `dataset.jsonl`).

---

## What is in this repo

| Component | Description |
|-----------|-------------|
| **Data pipeline** | GEE export → DIP/RGB → `dataset.jsonl` → frozen splits |
| **Zero-shot VLMs** | Four open-weight models on the **same 55 test patches** (`src/vlm_eval.py`) |
| **Fine-tune (09)** | LoRA on **Qwen2-VL-2B** using the **train** split only |
| **Results** | Per-model JSON + merged comparison tables under `results/` |

**On GitHub:** code, `dataset.jsonl`, `data/splits/`, result JSONs, summary figures.  
**Not on GitHub (local / Ada):** `data/processed_images/*.jpg` (~770 files, ~24 MB), heavy GeoTIFFs and `.npy` tensors.

---

## Dataset & splits

- **366 patches**, Condition C prompts in `dataset.jsonl` (v2).
- **Stratified split** on `candidate_class`, **seed 42**, 70% / 15% / 15%:

| Split | Patches | Use |
|-------|---------|-----|
| Train | 256 | LoRA fine-tune (notebook **09**) |
| Dev | 55 | Tuning / development (optional) |
| Test | 55 | All reported benchmark accuracies |

Manifest: `data/splits/split_manifest.json`, IDs in `data/splits/*_patches.csv`.

---

## Notebook workflow

| # | Notebook | Role |
|---|----------|------|
| 01 | `01_gee_extraction.ipynb` | GEE export (S2, SAR, CHIRPS, ERA5, NASADEM) |
| 02 | `02_dip_auto_classify.ipynb` | DIP, Otsu masks, RGB / tensors |
| 03 | `03_vlm_jsonl_builder.ipynb` | Build `dataset.jsonl` |
| 04 | `04_freeze_jsonl_splits.ipynb` | Freeze JSONL + write splits (`scripts/05_freeze_jsonl_and_splits.py`) |
| 05–08 | `05_vlm_llava15` … `08_vlm_blip2` | Zero-shot eval per VLM |
| 09 | `09_vlm_finetune_qwen.ipynb` | LoRA fine-tune Qwen2-VL (GPU) |

---

## Zero-shot benchmark (test split, n=55)

Merged table: **`results/vlm_combo_comparison.json`**.  
Summary figure: **`results/result2.png`** (regenerate with `python scripts/generate_result_figures.py`).

| Rank | Model | Test accuracy |
|------|--------|----------------|
| 1 | Qwen2-VL 2B (laptop HF) | **56.4%** |
| 2 | PaliGemma 3B | 36.4% |
| 3 | LLaVA 1.5 (0.5B interleave) | 32.7% |
| 4 | BLIP-2 OPT 2.7B | 32.7% |

Metric: parsed class from model text vs `candidate_class`.  
Notebooks **05–08** use `LAPTOP=True` (smaller checkpoints) and shared `eval_combo()` in `src/vlm_eval.py`.

Pipeline overview: **`results/result1.png`**.

---

## Fine-tuning (notebook 09 / Ada)

**Base model:** `Qwen/Qwen2-VL-2B-Instruct`  
**Combo id:** `qwen_vl_finetuned_ecoexplain`  
**Code:** `src/vlm_finetune.py`, `scripts/run_finetune_qwen.py`

```bash
# Local / GPU machine (after JPGs + dataset.jsonl + splits exist)
pip install torch torchvision transformers accelerate peft bitsandbytes
python scripts/run_finetune_qwen.py --train --eval --merge
```

**IIIT Ada (Slurm):**

```bash
cd ~/EcoExplain-VLM
sbatch scripts/ada_slurm_qwen09_smoke.sh   # quick check
sbatch scripts/ada_slurm_qwen09.sh         # full train + test eval
```

Use `source ~/venvs/ecoexplain/bin/activate` in the Slurm scripts (standalone Python 3.10 + venv on Ada).  
Outputs:

- `results/checkpoints/qwen2vl_lora_ecoexplain/` (gitignored)
- `results/vlm_combo_qwen_vl_finetuned_ecoexplain.json`
- `results/vlm_combo_comparison_with_finetune.json` (after `--merge`)

---

## Repository layout

```
├── configs/                 # pilot ROI / dates
├── data/
│   ├── splits/              # train / dev / test patch IDs
│   ├── processed_images/    # T0/T1 JPGs (local only)
│   └── raw_tiffs/           # GeoTIFFs (local only)
├── dataset.jsonl            # one row per patch (Condition C + labels)
├── notebooks/               # 01–09
├── scripts/
│   ├── 01_export_gee.py … 05_freeze_jsonl_and_splits.py
│   ├── run_finetune_qwen.py
│   ├── ada_slurm_qwen09*.sh
│   └── generate_result_figures.py
├── src/
│   ├── vlm_eval.py          # eval, parse_class, compare_reports
│   ├── vlm_combos.py        # model registry
│   └── vlm_finetune.py      # LoRA train/eval
└── results/
    ├── vlm_combo_*.json
    ├── vlm_combo_comparison.json
    ├── result1.png / result2.png
```

---

## Environment (laptop eval)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# VLM notebooks also need (uncomment in requirements.txt):
# torch torchvision transformers accelerate peft bitsandbytes
```

Use the project **`.venv`** kernel in Jupyter. Run **one** VLM notebook at a time on ~16 GB RAM; set `LAPTOP=True` and `PURGE_AFTER=True` in **05–08** to limit disk use.

---

## Data pipeline quick start (from scratch)

```bash
python scripts/01_export_gee.py
python scripts/03_build_dataset.py
python scripts/05_freeze_jsonl_and_splits.py
python scripts/04_validate.py
```

See notebooks **01–04** for the interactive path. Google Drive OAuth: `credentials.json` / `token.json` (never commit).

---

## Citation & contact

Independent Study — EcoExplain-VLM, Kanha pilot.  
GitHub: [Rama-Vaibhav/EcoExplain-VLM](https://github.com/Rama-Vaibhav/EcoExplain-VLM).
