#!/bin/bash
# Draft 1 review, sir's Major 5.4: the balanced 16-patch class LoRA with DIFFERENT balanced sample
# selections (train seeds), random split, evidence-only at train and test. One adapter per seed.
#   sbatch scripts/ada_slurm_qwen_seed_sweep.sh            (seeds 1 2 3 4 5)
#   SEEDS="7 11" sbatch scripts/ada_slurm_qwen_seed_sweep.sh

#SBATCH --job-name=eco-q-seeds
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-qwen-seed-sweep-%j.log
#SBATCH --error=results/slurm-qwen-seed-sweep-%j.log

set -euo pipefail
exec 2>&1
REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
pip freeze > "results/ada_pip_freeze_${SLURM_JOB_ID:-local}.txt"

for s in ${SEEDS:-1 2 3 4 5}; do
  echo "== seed $s $(date)"
  python scripts/run_finetune_qwen.py --train --eval \
    --max-train 16 --epochs 1 --balanced-train --train-seed "$s" \
    --eval-prompt evidence_only \
    --adapter-dir "results/checkpoints/qwen2vl_lora_seed$s" --out-tag "_seed$s"
done
echo "Seed sweep done $(date)"
