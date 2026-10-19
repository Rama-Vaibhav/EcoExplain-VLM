# EcoExplain-VLM

Independent Study benchmark: **forest vegetation change** at **Kanha Tiger Reserve**, matched dry season **January–February 2022 (T0) vs 2023 (T1)**.

Each patch is two RGB chips plus multimodal evidence. The label is **`stable` | `decline` | `increase`** (`candidate_class` in `dataset.jsonl`).

**GitHub:** [Rama-Vaibhav/EcoExplain-VLM](https://github.com/Rama-Vaibhav/EcoExplain-VLM) (`main`)  
**Local:** `ecoexplain_vlm_dataset/`  
**Ada:** `~/EcoExplain-VLM` (`ssh cprv67ab97@ada-gw1`)

This README is the source of truth for numbers a report should cite. Do **not** mix rows from different tables.

---

## 1. What this repo contains

| Piece | What it is |
|-------|------------|
| Data pipeline | Google Earth Engine → DIP/RGB → `dataset.jsonl` → frozen splits |
| Zero-shot VLMs | Four open models, **same 55 test patches** (`src/vlm_eval.py`) |
| Class LoRA | Qwen2-VL-2B, train split, Ada (`src/vlm_finetune.py`) |
| Explanations | Compact evidence table, JSON/short answers, faithfulness scoring |
| Explain LoRA | 2B teacher filters train justifications; 2B LoRA on those targets |
| Results | JSON under `results/` and `results/explain/` |

**On GitHub:** code, `dataset.jsonl`, `data/splits/`, `data/explain_targets/train_targets.jsonl`, result JSONs.  
**Not on GitHub:** `data/processed_images/*.jpg`, GeoTIFFs, `.npy`, LoRA checkpoints.

---

## 2. Dataset and splits

- **366 patches** (~640 m), Condition C cards in `dataset.jsonl` (v2).
- Sources: **Sentinel-2** (optical + NDVI/EVI/NDMI/NBR), **Sentinel-1** SAR, **CHIRPS** rainfall, **ERA5-Land** temperature and soil moisture, **NASADEM** terrain.
- Stratified on `candidate_class`, **seed 42**, 70 / 15 / 15:

| Split | n | Use |
|-------|---|-----|
| Train | 256 | LoRA |
| Dev | 55 | Early stopping / selection |
| Test | 55 | **All reported accuracies** |

Manifest: `data/splits/split_manifest.json`. IDs: `data/splits/*_patches.csv`.

**Condition C card:** images + measured evidence text. The full prompt **ends with a pipeline summary that includes `candidate_class=`** (same field as the gold label). Zero-shot 56.4% and class LoRA 67.3% use that full card. `evidence_only` / `minimal` / compact-evidence prompts **strip that line**. State this in Methods.

---

## 3. Notebooks

| # | Notebook | Role |
|---|----------|------|
| 01 | `01_gee_extraction.ipynb` | GEE export (S2, SAR, CHIRPS, ERA5, NASADEM) |
| 02 | `02_dip_auto_classify.ipynb` | DIP, Otsu, RGB / tensors |
| 03 | `03_vlm_jsonl_builder.ipynb` | Build `dataset.jsonl` |
| 04 | `04_freeze_jsonl_splits.ipynb` | Freeze JSONL + splits |
| 05–08 | `05_vlm_llava15` … `08_vlm_blip2` | Zero-shot eval (`LAPTOP=True`) |
| 09 | `09_vlm_finetune_qwen.ipynb` | Class LoRA (run via Ada scripts) |

---

## 4. Numbers to report (test n = 55 unless noted)

Majority class on test is `increase` at **34.5%** (19/55).

### 4.1 Primary classification — Condition C, one-word, full card

File: `results/vlm_combo_comparison.json`

| Model | Accuracy | Notes |
|-------|----------|--------|
| **Qwen2-VL-2B** | **56.4%** (31/55) | Main zero-shot result |
| PaliGemma 3B | 36.4% | Often `decline` |
| LLaVA-Interleave 0.5B | 32.7% | Always `decrease` |
| BLIP-2 OPT 2.7B | 32.7% | Prompt echo → parser `decline` |

**Inputs were not equal:** Qwen and LLaVA received both images and the full card; PaliGemma both images and one NDVI line; BLIP-2 one stitched image and no card. Notebooks 05–08. Equal-input and leak-free reruns: §4.7.

### 4.2 Class-only LoRA (Qwen2-VL-2B)

File: `results/vlm_combo_qwen_vl_finetuned_ecoexplain.json` (smoke)

| Run | Accuracy | How to describe it |
|-----|----------|-------------------|
| Ada **2722446**, 16 train, 1 epoch, full card | **67.3%** | Old run: first 16 train IDs (9 decline / 6 stable / 1 increase), full card at train and test; never predicted `stable`. Superseded by the leak-free run in §4.7 |
| Ada 2690 / 2792, ~256 train, 2–3 epochs, full card | **100%** | Label leak / copying `candidate_class`. **Do not cite as vision accuracy.** |
| Ada 2984 / 3054, 128 train, images-only test | **32.7%** | Train/test protocol mismatch |

### 4.3 Structured cite-only explanations (Mac, 6 Oct 2026)

Files: `results/explain/explain_structured_*.json`, `results/explain/explanation_scores.json`

| Model | Explain-mode class acc. | Correct numeric claims |
|-------|-------------------------|------------------------|
| Qwen2-VL-2B | 63.6% (35/55) | **75.8%** |
| LLaVA 0.5B | 45.1% (23/51 answered) | **71.1%** |

Class often follows the stated NDVI ±0.02 rule (rule alone = 63.6%). This does **not** replace §4.1.

### 4.4 Legacy free-form explanations

| Model | n | Class acc. | Claim correctness |
|-------|---|------------|-------------------|
| Qwen2-VL-2B | **3** only | 3/3 | 43.8% of 16 claims |
| LLaVA 0.5B | 55 | 29.1% | 56.7% |
| PaliGemma 3B | 55 | 33.3% of 36 parsed | almost no claims |
| BLIP-2 | 55 | 32.7% | repeated the prompt |

### 4.5 Explain LoRA on Ada (16–17 Oct 2026) — **pilot**

Teacher could **not** be 7B (`Disk quota exceeded`). Teacher = **Qwen2-VL-2B**.

| Step | Job | Result |
|------|-----|--------|
| Filter train justifications | **24298** | **28 / 256** kept (`data/explain_targets/train_targets.jsonl`) |
| LoRA + explain eval | **24389** | Used **16 of 28** (because `--max-train 128` takes the first 128 train IDs) |
| Test | same | **61.8%** explain-mode class acc. (34/55); **50.5%** of checkable claims correct; 74.5% of explanations have ≥1 wrong claim |

Files: `results/explain/explain_qwen_vl_finetuned_ecoexplain.json`, `results/explain/target_filter_report.json`, `results/slurm-explain-targets-24298.log`, `results/slurm-qwen-explain-24389.log`.

**Do not put 61.8% in the same table as 56.4% / 67.3%.** Different prompt, leak-free evidence table, 16 JSON targets, 2B teaching 2B.

### 4.6 Other probes (appendix)

- `evidence_only` one-word: PaliGemma 36.4%, LLaVA 36.4%, BLIP-2 32.7% (`results/explain/oneword_*.json`). Qwen only 3 patches in that file.
- Closed-set log-prob + NDVI rule (Ada): LLaVA **60.0% calibrated** / 32.7% raw; PaliGemma and BLIP-2 stay ~32.7%. Raw 32.7% is always-`decline`.
- ΔNDVI ±0.02 rule vs labels: **63.6%**.

### 4.7 Protocol reruns (19–20 Oct 2026): label line removed, equal inputs

Commands: `docs/PROTOCOL_RERUN.md`. Files: `results/protocol/`.

| Run | Input | Accuracy | Notes |
|-----|-------|----------|-------|
| Qwen2-VL-2B zero-shot (laptop) | `evidence_only` | 32.7% (18/55) | `vlm_combo_comparison_evidence_only.json` |
| Qwen2-VL-2B zero-shot (laptop) | `compact` (same numeric table for every model) | 32.7% (18/55) | answered `stable` for all 55; per-model file `vlm_combo_qwen_vl_custom_mlp_qwen.json` holds this run |
| LLaVA-Interleave 0.5B (laptop) | `compact` | **43.6%** (24/55) | 43 `decline`, 12 `increase`, never `stable` |
| PaliGemma 3B | `compact` | — | gated model, not loaded on the laptop |
| BLIP-2 OPT 2.7B | `compact` | — | 0/55 answers parsed after the anti-echo parser fix |
| Qwen2-VL-2B class LoRA, Ada **29575** | balanced 16 train (6 decline / 5 stable / 5 increase), `evidence_only` at train and test, 1 epoch | **36.4%** (20/55) | 53 `stable` answers: stable 18/18, decline 2/18, increase 0/19 |

Parser: `parse_class` now strips echoed instructions, so BLIP-2's echo no longer counts as `decline`.
Spatial split: `data/splits_spatial/` (whole lon/lat blocks held out; 266 / 46 / 54), written by `scripts/write_spatial_splits.py`; not yet evaluated.

---

## 5. How to run

### Laptop (16 GB, one model at a time)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# also: torch torchvision transformers accelerate peft bitsandbytes
```

Jupyter kernel = this `.venv`. Notebooks 05–08: `LAPTOP=True`, `PURGE_AFTER=True`.

Leak check (must print 0):

```bash
python scripts/check_no_leak.py
```

Explanations:

```bash
python scripts/run_explain.py --dry-run --limit 3
python scripts/run_explain.py --mode explain --prompt-style structured --combo qwen_vl_custom_mlp_qwen
python scripts/score_explanations.py
python scripts/make_explanation_report.py
```

### Ada class LoRA

```bash
cd ~/EcoExplain-VLM
source ~/venvs/ecoexplain/bin/activate
sbatch --mem=16G scripts/ada_slurm_qwen09_smoke.sh
sbatch --mem=16G scripts/ada_slurm_qwen09.sh
```

Do **not** use `--mem=32G` (`QOSMaxMemoryPerUser`). PyTorch writes to stderr; logs are `results/slurm-qwen09-<JOBID>.log`.

### Ada explain LoRA (after JPGs + splits exist)

```bash
# 1) teacher filter (2B; 7B will fill the home quota)
sbatch --mem=16G scripts/ada_slurm_explain_targets.sh
# wait for: survived N / 256  and  data/explain_targets/train_targets.jsonl

# 2) student LoRA
sbatch --mem=16G scripts/ada_slurm_qwen_explain.sh
```

HF download on Ada: `export HF_HUB_DISABLE_XET=1`. If disk fills, delete  
`~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct`  
and **keep** the 2B cache.

---

## 6. Core code

| Path | Role |
|------|------|
| `src/vlm_combos.py` | Model registry, `encoder_text`, `evidence_only`, leak tokens |
| `src/vlm_eval.py` | `eval_combo`, `parse_class`, `parse_explanation`, `collate` |
| `src/vlm_evidence.py` | Compact numeric table (no class names) |
| `src/explain_metrics.py` | Claim directions vs measured deltas, ROUGE-L, distinct-2 |
| `src/vlm_finetune.py` | Class LoRA + `--target explain` |
| `src/vlm_closedset.py` | Log-prob closed-set scoring |
| `scripts/run_finetune_qwen.py` | CLI (`--eval-prompt evidence_only`, `--target explain`) |
| `scripts/build_explain_targets.py` | Teacher filter (`--teacher`, default 2B) |
| `scripts/run_explain.py` | Zero-shot explain / oneword |
| `scripts/score_explanations.py` | Tables in `results/explain/` |
| `scripts/ada_slurm_*.sh` | Slurm |

`--target explain` **must not** train on the template `explanation` field in JSONL.

---

## 7. Repository layout

```
├── dataset.jsonl
├── data/splits/
├── data/explain_targets/train_targets.jsonl   # 28 filtered 2B justifications
├── data/processed_images/                     # local / Ada only
├── notebooks/01–09
├── scripts/   run_finetune_qwen.py, run_explain.py, ada_slurm_*.sh, …
├── src/       vlm_eval.py, vlm_finetune.py, vlm_evidence.py, explain_metrics.py
└── results/
    ├── vlm_combo_comparison.json
    ├── vlm_combo_qwen_vl_finetuned_ecoexplain.json
    ├── explain/explain_*.json, explain_structured_*.json, oneword_*.json
    ├── explain/explain_qwen_vl_finetuned_ecoexplain.json
    ├── slurm-explain-targets-24298.log
    └── slurm-qwen-explain-24389.log
```

Checkpoints: `results/checkpoints/` (gitignored).

---

## 8. Data pipeline from scratch

```bash
python scripts/01_export_gee.py
python scripts/03_build_dataset.py
python scripts/05_freeze_jsonl_and_splits.py
python scripts/04_validate.py
```

Interactive path: notebooks 01–04. Drive OAuth: `credentials.json` / `token.json` — never commit.

---

## 9. What a report should and should not say

**Say:** Qwen2-VL-2B is the strongest **zero-shot Condition C** model at **56.4%**. Class LoRA on 16 patches reached **67.3%** with the full card and collapsed `stable` into `decline`. Structured prompts raise **claim correctness** to ~71–76%. Explain LoRA is a **16-example 2B-on-2B pilot** at **61.8%** explain-mode accuracy and **50.5%** claim correctness.

**Do not say:** 100% LoRA is a vision result; 61.8% beats 56.4% on the same task; 60% closed-set LLaVA is zero-shot; Qwen explain was run on 55 patches in legacy mode (it was 3).

---

## Citation

Independent Study — EcoExplain-VLM, Kanha Tiger Reserve (T0 2022 / T1 2023).  
GitHub: [Rama-Vaibhav/EcoExplain-VLM](https://github.com/Rama-Vaibhav/EcoExplain-VLM).
