#!/bin/bash
# Quick GPU smoke test (~15–45 min): 16 train patches, 1 epoch, still evals 55 test.
#   sbatch scripts/ada_slurm_qwen09_smoke.sh

#SBATCH --job-name=eco-q09-smoke
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-qwen09-smoke-%j.log
#SBATCH --error=results/slurm-qwen09-smoke-%j.err
##SBATCH --partition=gpu

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
# module load python/3.10 cuda/12.1
source ~/venvs/ecoexplain/bin/activate
python -V
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

python scripts/run_finetune_qwen.py --train --eval --max-train 16 --epochs 1
echo "Smoke done $(date)"
