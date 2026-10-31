#!/usr/bin/env bash
# X8 (31 Oct 2026): matched zero-shot control for the spatial-split class LoRA (Ada job 50743).
# Same model, evidence-only card, 384 px, spatial test set (54). Mac, MPS, fp16.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
echo "== X8 $(date)"; .venv/bin/python -W ignore scripts/run_protocol_fair_eval.py --phase qwen-leakfree --laptop --max-side 384 \
  --splits-dir data/splits_spatial --results-subdir results/protocol/spatial_evidence_only --tag _384
echo "== done $(date)"
