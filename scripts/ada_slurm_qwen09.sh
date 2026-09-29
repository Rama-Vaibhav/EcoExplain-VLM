#!/bin/bash
# Full LoRA fine-tune (notebook 09) on Ada — submit from repo root:
#   mkdir -p results && sbatch scripts/ada_slurm_qwen09.sh
#
# IIIT Ada (edit if sbatch rejects flags — see http://hpc.iiit.ac.in/wiki):
#   account: research   qos: low   user: your Ada login

#SBATCH --job-name=eco-qwen09
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-qwen09-%j.log
#SBATCH --error=results/slurm-qwen09-%j.err
## If required on your cluster, uncomment ONE partition line:
##SBATCH --partition=gpu

set -euo pipefail

# Repo root = directory where you ran sbatch (recommended: ~/EcoExplain-VLM)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO="${SLURM_SUBMIT_DIR}"
else
  REPO="${REPO:-$HOME/EcoExplain-VLM}"
fi
cd "$REPO"

# --- Modules (uncomment/edit per Ada User Guide) ---
# module purge
# module load python/3.10
# module load cuda/12.1

# --- Python env: micromamba (Ada login node is Python 3.6 / old glibc) ---
source ~/venvs/ecoexplain/bin/activate
python -V

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== EcoExplain 09 | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "host=$(hostname) pwd=$(pwd)"
echo "python=$(which python) $(python -V)"
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# --- Data preflight ---
if [[ ! -f dataset.jsonl ]]; then
  echo "ERROR: missing dataset.jsonl in $REPO"
  exit 1
fi
if [[ ! -d data/splits ]]; then
  echo "ERROR: missing data/splits/"
  exit 1
fi
N_JPG=$(find data/processed_images -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
echo "jpg_count=$N_JPG (expect hundreds of T0/T1 files)"
if [[ "${N_JPG:-0}" -lt 100 ]]; then
  echo "ERROR: too few JPGs under data/processed_images/ — extract ecoexplain_jpg.tar.gz"
  exit 1
fi

mkdir -p results

# --- Train 256 patches, eval test 55, merge comparison JSON ---
python scripts/run_finetune_qwen.py --train --eval --merge

echo "=== Finished $(date) ==="
echo "Finetuned report:"
python - <<'PY'
import json
from pathlib import Path
p = Path("results/vlm_combo_qwen_vl_finetuned_ecoexplain.json")
if p.exists():
    r = json.loads(p.read_text())
    acc = r.get("accuracy")
    pct = f"{100*acc:.1f}%" if acc is not None else "n/a"
    print(f"  combo_id={r.get('combo_id')} accuracy={pct} n_eval={r.get('n_eval')} parsed={r.get('n_parsed_class')}")
else:
    print("  (missing results/vlm_combo_qwen_vl_finetuned_ecoexplain.json)")
cmp = Path("results/vlm_combo_comparison_with_finetune.json")
if cmp.exists():
    c = json.loads(cmp.read_text())
    print(f"  best_combo={c.get('best_combo_id')} best_acc={c.get('best_accuracy_pct')}")
PY
