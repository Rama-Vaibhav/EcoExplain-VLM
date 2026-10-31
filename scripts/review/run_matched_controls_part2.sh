#!/usr/bin/env bash
# Second batch (31 Oct 2026): put both arms of the primary leakage test in today's environment.
# The current transformers loads Qwen2-VL's image processor in "fast" mode by default, so the
# September full-card run may not match today's set-up exactly.
#   X7  Qwen full card (with summary line), random test, 336 px  -> pairs with X1 (evidence-only, 336 px)
#   X6  Qwen compact, random test, 336 px, re-run               -> reproduction check for Qwen (old: 18/55, all stable)
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
OUT=results/protocol

echo "== X7 $(date)"; $PY -W ignore scripts/run_protocol_fair_eval.py --phase fair-four --laptop \
  --combo qwen_vl_custom_mlp_qwen --text-mode full --results-subdir $OUT/random_full_card --tag _336_rerun
echo "== X6 $(date)"; $PY -W ignore scripts/run_protocol_fair_eval.py --phase fair-four --laptop \
  --combo qwen_vl_custom_mlp_qwen --results-subdir $OUT/repro_gate --tag _qwen_repro336
echo "== done $(date)"
