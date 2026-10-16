#!/bin/bash
# LoRA-tune Qwen2-VL-2B on filtered explanation targets, then explain-mode test eval.
# Requires data/explain_targets/train_targets.jsonl from ada_slurm_explain_targets.sh.
# Writes results/explain/ only (does not replace results/vlm_combo_*.json).
#
#   cd ~/EcoExplain-VLM && sbatch scripts/ada_slurm_qwen_explain.sh

#SBATCH --job-name=eco-xtrain
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-qwen-explain-%j.log
#SBATCH --error=results/slurm-qwen-explain-%j.log

set -euo pipefail
exec 2>&1

REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== explain LoRA | job ${SLURM_JOB_ID:-local} | $(date) ==="
[[ -f data/explain_targets/train_targets.jsonl ]] || { echo "ERROR: missing train_targets.jsonl"; exit 1; }
python scripts/run_finetune_qwen.py --train --eval --target explain --max-train 128 --epochs 1
python scripts/score_explanations.py
echo "=== Finished $(date) ==="
