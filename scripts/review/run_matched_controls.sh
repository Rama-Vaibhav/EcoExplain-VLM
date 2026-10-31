#!/usr/bin/env bash
# Matched-control and reproduction runs for the Draft 1 review (31 Oct 2026). Mac, MPS, fp16, 2B Qwen / 0.5B LLaVA.
#   X5  reproduction gate: LLaVA compact, random test, 336 px  -> compare with git 6dd2088 (24/55)
#   X1  Qwen evidence-only, random test, 336 px                -> paired with the 336 px full-card run (primary leakage test)
#   X2  Qwen evidence-only, random test, 384 px                -> matched zero-shot control for the class LoRA (384 px)
#   X3  Qwen legacy JSON explanation, random test, 384 px       -> matched zero-shot control for the explanation LoRA (384 px)
#   X4  Qwen structured explanation, 512 new tokens, 336 px    -> does the 256-token truncation change claim consistency?
# Every run writes its own file; nothing is overwritten (src/vlm_eval.py::no_clobber_path).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
OUT=results/protocol

echo "== X5 $(date)"; $PY -W ignore scripts/run_protocol_fair_eval.py --phase fair-four --laptop \
  --combo llava15_clip_mlp_vicuna --results-subdir $OUT/repro_gate --tag _repro336
echo "== X1 $(date)"; $PY -W ignore scripts/run_protocol_fair_eval.py --phase qwen-leakfree --laptop \
  --results-subdir $OUT/random_evidence_only --tag _336
echo "== X2 $(date)"; $PY -W ignore scripts/run_protocol_fair_eval.py --phase qwen-leakfree --laptop --max-side 384 \
  --results-subdir $OUT/random_evidence_only --tag _384
echo "== X3 $(date)"; $PY -W ignore scripts/run_explain.py --mode explain --prompt-style legacy \
  --combo qwen_vl_custom_mlp_qwen --max-side 384 --tag _384
echo "== X4 $(date)"; $PY -W ignore scripts/run_explain.py --mode explain --prompt-style structured \
  --combo qwen_vl_custom_mlp_qwen --max-new-tokens 512 --tag _tok512
echo "== done $(date)"
