# Protocol fix reruns

After the code changes for **leak-free eval**, **balanced LoRA**, **compact fair text**, **parser anti-echo**, and **spatial splits**, use these commands from the repo root (`ecoexplain_vlm_dataset/`).

## 0. Sync to Ada (from your Mac)

**Do not** `rsync` the whole `data/processed_images/` folder: it is ~16 GB of `.npy` tensors. LoRA/VLM only need the **~24 MB of `.jpg` chips**.

```bash
cd "/Users/vaibhav/Desktop/Independent Study/is work/DataSet Extractor/ecoexplain_vlm_dataset"
export ADA_HOST=cprv67ab97@ada-gw1
./scripts/sync_ada_jpg_only.sh
```

If Ada still reports `Disk quota exceeded`, SSH in and free space **before** syncing again:

```bash
ssh cprv67ab97@ada-gw1
du -sh ~/EcoExplain-VLM/data/processed_images ~/.cache/huggingface 2>/dev/null
find ~/EcoExplain-VLM/data/processed_images -name '*.npy' -delete
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct   # keep 2B cache
```

## 1. Spatial split (no GPU, ~5 s)

```bash
python scripts/write_spatial_splits.py
# -> data/splits_spatial/{train,dev,test}_patches.csv
```

Keep **`data/splits/`** as the primary 256/55/55 table for the paper. Use **`data/splits_spatial/`** as a second, leakage-reduced test.

## 2. Leak check

```bash
python scripts/check_no_leak.py
```

## 3. Qwen zero-shot, label line removed, all 55 test patches

**Laptop** (one model, ~30–90 min):

```bash
source .venv/bin/activate
python scripts/run_protocol_fair_eval.py --phase qwen-leakfree --laptop --purge-after --max-eval 55
```

**Output:** `results/protocol/vlm_combo_comparison_evidence_only.json`  
and `results/protocol/vlm_combo_qwen_vl_custom_mlp_qwen.json` (per-model detail).

## 4. Four models, same compact evidence (fair text)

```bash
python scripts/run_protocol_fair_eval.py --phase fair-four --laptop --purge-after --max-eval 55
```

**Output:** `results/protocol/vlm_combo_comparison_compact.json`

On spatial test only:

```bash
python scripts/run_protocol_fair_eval.py --phase fair-four --laptop --purge-after \
  --splits-dir data/splits_spatial --max-eval 55
```

## 5. Balanced class LoRA (16 patches, leak-free) — Ada

```bash
ssh cprv67ab97@ada-gw1
cd ~/EcoExplain-VLM && source ~/venvs/ecoexplain/bin/activate
sbatch --mem=16G scripts/ada_slurm_qwen_balanced_smoke.sh
tail -f results/slurm-qwen-balanced-smoke-<JOBID>.log
```

Log should print `train class mix: {'decline': 6, 'stable': 5, 'increase': 5}` (or 5/5/6).

**Output:** `results/vlm_combo_qwen_vl_finetuned_ecoexplain.json` (overwrite or copy to `results/protocol/` before rerun).

Eval uses `--eval-prompt evidence_only` (no `candidate_class` line).

## 6. Optional: balanced 48 (16 per class), 1 epoch

```bash
python scripts/run_finetune_qwen.py --train --eval --eval-dev \
  --max-train 48 --epochs 1 --balanced-train --eval-prompt evidence_only
```

## 7. Update the report

| Old headline | Replace with |
|--------------|----------------|
| 56.4% Qwen (full card) | Keep as **legacy / leaky** row or appendix |
| 67.3% class LoRA | **Balanced 16 + evidence_only** from new Ada log |
| Four-model table | **`results/protocol/vlm_combo_comparison_compact.json`** |
| Primary Qwen zero-shot | **`results/protocol/vlm_combo_comparison_evidence_only.json`** |
| Spatial robustness | Compact fair eval on `splits_spatial` test |

Parser fix applies automatically to new runs (BLIP echo no longer forced to “decline”).

## 8. Quick parser smoke test (no GPU)

```bash
python -c "from src.vlm_eval import parse_class; \
print(parse_class('Side by side: left forest patch T0, right forest patch T1. One word only — decline, increase, or stable:'))"
# expect None or empty, not 'decline'
```
