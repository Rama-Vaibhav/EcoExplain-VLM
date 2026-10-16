#!/bin/bash
# Structured explain-mode eval on Ada (cite-only ALLOWED_FACTS prompts).
# Does not need your Mac on. Writes results/explain/explain_structured_*.json
#
# On Ada (after syncing repo + JPGs + dataset.jsonl):
#   cd ~/EcoExplain-VLM && mkdir -p results/explain
#   sbatch scripts/ada_slurm_explain_structured.sh
#
# Optional env (export before sbatch or pass via --export):
#   EXPLAIN_COMBOS="qwen_vl_custom_mlp_qwen llava15_clip_mlp_vicuna"
#   EXPLAIN_LIMIT=5          # smoke: first N test patches only
#   EXPLAIN_STYLE=structured # default; use legacy to compare old prompts
#
# Sync to Ada (do NOT rsync .npy — fills disk quota). From Mac:
#   ./scripts/sync_ada_jpg_only.sh
#
# Pull results back to Mac:
#   scp 'cprv67ab97@ada-gw1:~/EcoExplain-VLM/results/explain/explain_structured_*' results/explain/
#   scp 'cprv67ab97@ada-gw1:~/EcoExplain-VLM/results/explain/explanation_scores.*' results/explain/
#   scp 'cprv67ab97@ada-gw1:~/EcoExplain-VLM/results/slurm-explain-structured-*.log' results/

#SBATCH --job-name=eco-explain
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-explain-structured-%j.log
#SBATCH --error=results/slurm-explain-structured-%j.log

set -euo pipefail
exec 2>&1

REPO="${SLURM_SUBMIT_DIR:-$HOME/EcoExplain-VLM}"
cd "$REPO"
source ~/venvs/ecoexplain/bin/activate
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

STYLE="${EXPLAIN_STYLE:-structured}"
COMBOS="${EXPLAIN_COMBOS:-qwen_vl_custom_mlp_qwen llava15_clip_mlp_vicuna paligemma_siglip_linear_gemma blip2_clip_qformer_opt}"
LIMIT_ARG=()
if [[ -n "${EXPLAIN_LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$EXPLAIN_LIMIT")
fi

echo "=== structured explain | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "host=$(hostname) pwd=$(pwd) style=$STYLE"
python -V
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

[[ -f dataset.jsonl ]] || { echo "ERROR: missing dataset.jsonl"; exit 1; }
[[ -d data/splits ]] || { echo "ERROR: missing data/splits"; exit 1; }
N_JPG=$(find data/processed_images -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
echo "jpg_count=$N_JPG"
[[ "${N_JPG:-0}" -ge 100 ]] || { echo "ERROR: too few JPGs under data/processed_images"; exit 1; }

mkdir -p results/explain
if [[ -f scripts/check_no_leak.py ]]; then
  python scripts/check_no_leak.py || echo "WARN: check_no_leak failed"
else
  echo "skip check_no_leak.py (not synced; optional)"
fi

for combo in $COMBOS; do
  echo "--- explain $combo | style=$STYLE | $(date) ---"
  python scripts/run_explain.py \
    --mode explain \
    --prompt-style "$STYLE" \
    --no-laptop \
    --combo "$combo" \
    "${LIMIT_ARG[@]}"
done

echo "--- scoring ---"
python scripts/score_explanations.py

echo "=== outputs ==="
ls -la results/explain/explain_structured_*.json 2>/dev/null || ls -la results/explain/explain_*.json
echo "=== Finished $(date) ==="
