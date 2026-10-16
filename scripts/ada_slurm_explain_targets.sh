#!/bin/bash
# Build explanation targets on the train split.
# Ada home quota often cannot hold Qwen2-VL-7B, so this job uses 2B (already cached).
#   cd ~/EcoExplain-VLM && mkdir -p results data/explain_targets && sbatch scripts/ada_slurm_explain_targets.sh

#SBATCH --job-name=eco-xtarget
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-explain-targets-%j.log
#SBATCH --error=results/slurm-explain-targets-%j.log

set -euo pipefail
exec 2>&1

REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Xet downloader often dies on Ada with "Background writer channel closed".
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== explain targets | job ${SLURM_JOB_ID:-local} | $(date) ==="
python -c "import torch; print('cuda', torch.cuda.is_available())"
python scripts/build_explain_targets.py --split train --teacher Qwen/Qwen2-VL-2B-Instruct
echo "=== Finished $(date) ==="
python - <<'PY'
import json
from pathlib import Path
p = Path("results/explain/target_filter_report.json")
print(p.read_text() if p.exists() else "missing filter report")
PY
