#!/bin/bash
# Balanced 16-patch class LoRA, leak-free train+test (evidence_only), 1 epoch.
#   sbatch --mem=16G scripts/ada_slurm_qwen_balanced_smoke.sh

#SBATCH --job-name=eco-q-bal16
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-qwen-balanced-smoke-%j.log
#SBATCH --error=results/slurm-qwen-balanced-smoke-%j.log

set -euo pipefail
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1

python scripts/run_finetune_qwen.py \
  --train --eval --eval-dev \
  --max-train 16 --epochs 1 \
  --balanced-train \
  --eval-prompt evidence_only

echo "Balanced leak-free smoke done $(date)"
