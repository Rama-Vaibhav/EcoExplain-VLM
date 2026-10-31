#!/bin/bash
# Draft 1 review, sir's Major 3.1: balanced 16-patch class LoRA trained on the SPATIAL train blocks
# and tested on the spatial test blocks (54 patches). Same recipe as job 29575 otherwise.
# Spatial dev has only 2 increase patches, so it is evaluated for the record but not used to choose.
#   sbatch scripts/ada_slurm_qwen_spatial_lora.sh

#SBATCH --job-name=eco-q-spatial
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-qwen-spatial-lora-%j.log
#SBATCH --error=results/slurm-qwen-spatial-lora-%j.log

set -euo pipefail
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
pip freeze > "results/ada_pip_freeze_${SLURM_JOB_ID:-local}.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv > "results/ada_gpu_${SLURM_JOB_ID:-local}.txt" || true

python scripts/run_finetune_qwen.py --train --eval --eval-dev \
  --max-train 16 --epochs 1 --balanced-train --train-seed 42 \
  --splits-dir data/splits_spatial --eval-prompt evidence_only \
  --adapter-dir results/checkpoints/qwen2vl_lora_spatial --out-tag _spatial
echo "Spatial LoRA done $(date)"
