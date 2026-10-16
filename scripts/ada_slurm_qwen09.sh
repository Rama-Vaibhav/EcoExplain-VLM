#!/bin/bash
# EcoExplain notebook 09 — anti-overfit LoRA on Ada.
# CLI defaults (do not assume train 256 / 3 epochs): max-train 128, 1 epoch,
# LoRA r=8, lr 5e-5, weight decay 0.05, eval prompt = minimal (no spectral text).
# Training still uses full Condition C, which includes "candidate_class=...".
# Submit from repo root after this script and src/vlm_*.py are on Ada:
#   cd ~/EcoExplain-VLM && mkdir -p results && sbatch scripts/ada_slurm_qwen09.sh
#
# If sbatch fails with "Invalid account or account/partition":
#   1) Try PROFILE=B below (no account/qos)
#   2) Email HPC — association may need renewal after ada-gw1

#SBATCH --job-name=eco-qwen09
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-qwen09-%j.log
#SBATCH --error=results/slurm-qwen09-%j.log

# PROFILE=B fallback — comment out the four lines above and uncomment these five:
##SBATCH --job-name=eco-qwen09
##SBATCH --partition=u22
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=16G
##SBATCH --time=12:00:00
##SBATCH --output=results/slurm-qwen09-%j.log
##SBATCH --error=results/slurm-qwen09-%j.err

set -euo pipefail
# PyTorch / HF / tqdm write to stderr; merge into one log file.
exec 2>&1

REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"

source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== EcoExplain 09 | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "host=$(hostname) pwd=$(pwd)"
python -V
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

[[ -f dataset.jsonl ]] || { echo "ERROR: missing dataset.jsonl"; exit 1; }
[[ -d data/splits ]] || { echo "ERROR: missing data/splits"; exit 1; }
N_JPG=$(find data/processed_images -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
echo "jpg_count=$N_JPG"
[[ "${N_JPG:-0}" -ge 100 ]] || { echo "ERROR: too few JPGs"; exit 1; }

mkdir -p results
echo "protocol: max-train=128 epochs=1 eval-prompt=minimal eval-dev=yes merge=yes"
python scripts/run_finetune_qwen.py --train --eval --eval-dev --merge \
  --max-train 128 --epochs 1 --eval-prompt minimal

echo "=== Finished $(date) ==="
python - <<'PY'
import json
from pathlib import Path
p = Path("results/vlm_combo_qwen_vl_finetuned_ecoexplain.json")
if p.exists():
    r = json.loads(p.read_text())
    acc = r.get("accuracy")
    pct = f"{100*acc:.1f}%" if acc is not None else "n/a"
    print(
        f"accuracy={pct} n_eval={r.get('n_eval')} "
        f"prompt_mode={r.get('prompt_mode')} eval_split={r.get('eval_split')}"
    )
else:
    print("missing results/vlm_combo_qwen_vl_finetuned_ecoexplain.json")
PY
