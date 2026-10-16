#!/bin/bash
# Closed-set rescore: LLaVA 0.5B, PaliGemma 3B, BLIP-2 OPT 2.7B.
# Does not train Qwen and does not overwrite results/vlm_combo_*.json.
#
#   cd ~/EcoExplain-VLM && mkdir -p results && sbatch scripts/ada_slurm_vlm_closedset.sh

#SBATCH --job-name=eco-closedset
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-closedset-%j.log
#SBATCH --error=results/slurm-closedset-%j.log

set -euo pipefail
exec 2>&1

REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"

source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== EcoExplain closed-set | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "host=$(hostname) pwd=$(pwd)"
python -V
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

[[ -f dataset.jsonl ]] || { echo "ERROR: missing dataset.jsonl"; exit 1; }
[[ -d data/splits ]] || { echo "ERROR: missing data/splits"; exit 1; }
N_JPG=$(find data/processed_images -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
echo "jpg_count=$N_JPG"
[[ "${N_JPG:-0}" -ge 100 ]] || { echo "ERROR: too few JPGs"; exit 1; }

mkdir -p results
python scripts/run_vlm_closedset.py --max-eval 55

echo "=== Finished $(date) ==="
python - <<'PY'
import json
from pathlib import Path
p = Path("results/vlm_closedset_comparison.json")
if not p.exists():
    raise SystemExit("missing results/vlm_closedset_comparison.json")
summary = json.loads(p.read_text())
print("qwen_zero_shot", summary.get("qwen_zero_shot_accuracy_pct"))
for row in summary.get("ranking", []):
    print(
        row["combo_id"],
        "calibrated", row.get("test_accuracy_calibrated_pct"),
        "raw", row.get("test_accuracy_raw"),
        "beats_56.4", row.get("beats_qwen_56_4"),
    )
for fail in summary.get("failures", []):
    print("FAILED", fail.get("combo_id"), fail.get("error"))
PY
