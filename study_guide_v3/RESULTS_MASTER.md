# EcoExplain-VLM: master results (2026-10-20)

Recomputed from the prediction files by `build_results_master.py`. Test set n = 55.
**Rule:** never compare rows from different sections without naming the protocol.

| Section | Run | Protocol | Accuracy | 95% CI | Macro-F1 | Correct per class (d / s / i) | Claims correct | Source |
|---|---|---|---|---|---|---|---|---|
| A legacy | Qwen2-VL-2B | images + full card (summary line with label) | 56.4% (31/55) | 43.3-68.6 | 0.466 | 18/18 / 13/18 / 0/19 | — | `results/vlm_combo_qwen_vl_custom_mlp_qwen.json` |
| A legacy | PaliGemma-3B | images + one NDVI line | 36.4% (20/55) | 24.9-49.6 | 0.271 | 15/18 / 0/18 / 5/19 | — | `results/vlm_combo_paligemma_siglip_linear_gemma.json` |
| A legacy | LLaVA-Interleave-0.5B | images + full card | 32.7% (18/55) | 21.8-45.9 | 0.164 | 18/18 / 0/18 / 0/19 | — | `results/vlm_combo_llava15_clip_mlp_vicuna.json` |
| A legacy | BLIP-2 OPT-2.7B | stitched image, no card (old parser read the echo) | 32.7% (18/55) | 21.8-45.9 | 0.164 | 18/18 / 0/18 / 0/19 | — | `results/vlm_combo_blip2_clip_qformer_opt.json` |
| A leak-free | Qwen2-VL-2B | evidence_only card (label line removed) | 32.7% (18/55) | 21.8-45.9 | — | — | — | `results/protocol/vlm_combo_comparison_evidence_only.json` |
| A leak-free | LLaVA-Interleave-0.5B | compact table (same for all models) | 43.6% (24/55) | 31.4-56.7 | 0.347 | 16/18 / 0/18 / 8/19 | — | `results/protocol/vlm_combo_llava15_clip_mlp_vicuna.json` |
| A leak-free | Qwen2-VL-2B | compact table | 32.7% (18/55) | 21.8-45.9 | 0.164 | 0/18 / 18/18 / 0/19 | — | `results/protocol/vlm_combo_qwen_vl_custom_mlp_qwen.json` |
| A leak-free | PaliGemma-3B | compact table | not loaded | — | — | — | — | `results/protocol/vlm_combo_paligemma_siglip_linear_gemma.json` |
| A leak-free | BLIP-2 OPT-2.7B | compact table (new anti-echo parser) | 0/55 parsed | — | — | — | — | `results/protocol/vlm_combo_blip2_clip_qformer_opt.json` |
| B class LoRA | Qwen2-VL-2B, balanced 16 (Ada 29575) | evidence_only at train and test | 36.4% (20/55) | 24.9-49.6 | 0.236 | 2/18 / 18/18 / 0/19 | — | `results/protocol/qwen_balanced_leakfree_lora.json` |
| B class LoRA (superseded) | Qwen2-VL-2B, first 16 by ID (Ada 2722446) | full card at train and test | 67.3% (37/55) | 54.1-78.2 | 0.556 | 18/18 / 0/18 / 19/19 | — | `results/vlm_combo_qwen_vl_finetuned_ecoexplain.json` |
| C explanation | Qwen2-VL-2B structured | label-free table + ALLOWED_FACTS + CLASS_RULE | 63.6% (35/55) | 50.4-75.1 | 0.611 | 17/18 / 13/18 / 5/19 | 75.8% | `results/explain/explain_structured_qwen_vl_custom_mlp_qwen.json` |
| C explanation | LLaVA-0.5B structured | same | 41.8% (23/55); 45.1% of 51 answered | 29.7-55.0 | 0.355 | 13/18 / 10/18 / 0/19 | 71.1% | `results/explain/explain_structured_llava15_clip_mlp_vicuna.json` |
| C explanation | Qwen2-VL-2B explanation LoRA (Ada 24389) | label-free table, free-form JSON | 61.8% (34/55) | 48.6-73.5 | 0.525 | 17/18 / 0/18 / 17/19 | 50.5% | `results/explain/explain_qwen_vl_finetuned_ecoexplain.json` |

## Baselines (no model)

- Majority class (always increase): 34.5% (19/55)
- ΔNDVI rule, fixed ±0.02: 63.6% (35/55)
- ΔNDVI rule, cut-offs fitted on train [-0.032, -0.0011]: 94.5% (52/55)
- LLaVA closed-set probe with dev-fitted bias: 60.0% (33/55), probe only (ada_review/)

Explanation teacher filter: 28/256 kept.
