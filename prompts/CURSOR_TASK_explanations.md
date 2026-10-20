# Task: add model-written justifications + a visual comparison report to EcoExplain-VLM

Context: this repo benchmarks vision-language models (VLMs) on forest change classification
(stable / decline / increase) for 366 patches in Kanha Tiger Reserve, T0 = Jan–Feb 2022,
T1 = Jan–Feb 2023. Gold label = `candidate_class` in `dataset.jsonl`. Test split = the 55 IDs in
`data/splits/test_patches.csv`. Current eval: `src/vlm_eval.py` (`collate`, `generate_one`,
`parse_class`, `eval_combo`), models in `src/vlm_combos.py`, fine-tuning in `src/vlm_finetune.py`.
Right now models only output one word and we only score accuracy. I want each model to also
**justify** its prediction in its own words, and a visual report I can show my professor.

Do NOT overwrite any existing file in `results/`. Write all new outputs to `results/explain/`.
Keep the same 55 test patch IDs, greedy decoding (`do_sample=False`), and the `LAPTOP=True` path
(I run on a 16 GB Mac, one model at a time).

---

## Part 0 — Fix label leakage (do this first)

`encoder_text()` in `src/vlm_combos.py` returns `text.condition_c_user_prompt`, which ends with a line
like `Inference for p00000: vegetation_trend=moderate_decline, burn/deforestation cue=low,
candidate_class=stable (...)`. That is the gold label, so the model can copy the answer. It is
present in all 366 patches. Also the analyst prose contains interpretive tags
(e.g. "Moderate canopy loss", `drier_second_window`, `steep_terrain`).

1. Add a function `clean_evidence_text(record) -> str` that removes the whole `Inference for ...`
   line and any `candidate_class`, `vegetation_trend`, `reference_class`, `partial_mismatch`
   tokens, and strips the trailing "Respond with class..." line.
2. Add a new `text_mode="evidence_only"` to `encoder_text()` that uses `clean_evidence_text`.
3. Add an assertion in `collate()`: for any text_mode other than the old `"full"`, raise if the input
   text contains `candidate_class` or the gold label string after `=`.
4. Add a short test script `scripts/check_no_leak.py` that runs over all 366 records and prints how many
   inputs still contain label tokens (must be 0).
5. In `src/vlm_finetune.py`, make sure both training and evaluation use `evidence_only` (or `minimal`),
   never `full`, and save `prompt_mode` into the result JSON.

## Part 1 — Ask models for a justification (zero-shot)

Add a new prompt mode `explain` in `src/vlm_eval.py` (keep the old one-word mode untouched for
comparison). Instead of the analyst paragraph, give the model a **compact numeric evidence table**
built from `dataset.jsonl` fields (no interpretive words), for example:

```
Evidence (T0 2022 -> T1 2023, patch means):
NDVI 0.662 -> 0.631 (Δ -0.031); 18.3% of pixels dropped by more than 0.05
EVI Δ +0.013 | NDMI Δ +0.005 | NBR Δ +0.023
Green cover in RGB 0.96 -> 0.77
SAR VV Δ -0.13 dB, VH Δ -0.10 dB
Rainfall Jan–Feb 55.4 -> 24.5 mm
Soil moisture anomaly +0.023 -> -0.037 | Temperature anomaly -1.5 -> +0.7 °C
Elevation 688 m, slope 21°
```

Prompt (adapt wording per model):

> Image 1 is a forest patch in Jan–Feb 2022, image 2 is the same patch in Jan–Feb 2023.
> First describe in one sentence what visibly changed between the two images.
> Then decide whether vegetation is stable, declined, or increased.
> Justify your decision in 2–3 sentences in your own words, citing the specific numbers that support it,
> and mention any evidence that contradicts it. If climate (rainfall, soil moisture, temperature)
> could explain the change, say so as a possibility, not a certainty. Do not repeat the table.
> Answer as JSON: {"visual_change": "...", "change_class": "stable|decline|increase",
> "justification": "...", "supporting_evidence": ["..."], "conflicting_evidence": ["..."],
> "possible_driver": "...", "confidence": 0.0-1.0}

- Set `max_new_tokens` to 256 for Qwen2-VL and LLaVA in `explain` mode.
- PaliGemma and BLIP-2 cannot follow long JSON instructions (see existing `PALIGEMMA_PROMPT_SUFFIX`,
  `BLIP2_PROMPT`). For them use a short prompt asking for `class: <word>. reason: <one sentence>`,
  `max_new_tokens` ~80, and parse leniently. Record that they used the short format.
- Write a robust parser `parse_explanation(text) -> dict`: try JSON (also inside ```json fences or
  with trailing text), then fall back to regex for class + remaining text as justification. Keep
  `parse_class` for the class so accuracy stays comparable.
- Save per model: `results/explain/explain_<combo_id>.json` with, per patch: id, gold, pred,
  correct, raw text, parsed fields, prompt_mode, seconds.
- Add a `--limit 3` dry run option so I can check output on 3 patches before the full 55.
- Also re-run the **old one-word mode with `evidence_only`** so I have a clean, leak-free accuracy
  number for each model.

## Part 2 — Score the justifications

New script `scripts/score_explanations.py` that reads `results/explain/explain_*.json` and
`dataset.jsonl` and computes per model:

1. **Accuracy** (pred vs gold).
2. **Faithfulness**: extract claims of the form <indicator> + <direction> from the justification with
   regex. Indicators: NDVI, EVI, NDMI, NBR, green cover/greenness/canopy, VV, VH/backscatter,
   rainfall/precipitation/drier/wetter, soil moisture, temperature/warmer/cooler. Directions:
   increase/rise/higher/gain vs decrease/drop/decline/lower/loss vs stable/unchanged/little change.
   Compare with the measured delta for that patch (use a small dead-band, e.g. |ΔNDVI| < 0.02 counts as
   "stable"; put thresholds in one dict at the top). Report: checkable claims per explanation,
   % correct claims, % explanations with at least one wrong claim (hallucination rate).
3. **Self-consistency**: does the justification's direction agree with the model's own predicted class?
4. **Originality / not template-copied**: ROUGE-L (or longest common token n-gram) between the
   justification and (a) the input evidence text and (b) the rule-based `explanation` field in
   `dataset.jsonl`. Also distinct-2 across all of a model's explanations and the share of
   near-duplicate explanations across patches (similarity > 0.9). High overlap = copying.
5. Save `results/explain/explanation_scores.json` and a CSV table, and print a markdown table.

## Part 3 — Visual report for my professor

New script `scripts/make_explanation_report.py`:

1. **Per-patch panels** (`results/explain/panels/<id>.png`, 300 dpi):
   - Top row: T0 image | T1 image | ΔNDVI map (from `evidence.delta_ndvi` .npy, diverging colormap
     centred at 0, with colorbar). Titles "2022 (before)", "2023 (after)", "ΔNDVI".
   - Header: patch id, gold label, and 4–5 key measured numbers.
   - Below: one block per model: model name, predicted label in a coloured badge
     (decline = red, increase = green, stable = grey) with ✓/✗ against gold, the justification
     text (word-wrapped, readable font size ≥ 9 pt), and its faithfulness score.
2. **Patch selection**: pick about 9 showcase patches from the test split: 3 per gold class,
   preferring a mix of "all models right", "models disagree", and "all models wrong". Also allow
   `--ids p00052,p00010` to choose manually.
3. **Combined outputs**: all panels in one PDF (`results/explain/explanation_report.pdf`) and one
   self-contained HTML gallery (`results/explain/explanation_report.html`) with images embedded.
4. **Summary figure** `results/explain/summary_scores.png`: grouped bars per model for accuracy
   (leaky one-word, clean one-word, explain mode), faithfulness, and the majority-class baseline
   (34.5%) as a dashed line. Legible labels, no overlapping text.

## Part 4 — Fine-tuning with justifications (later, on Ada; implement but don't run)

Do NOT use the rule-based `explanation` field as the training target (it's a template, the model
would just learn the template). Instead:
1. Generate candidate justifications for train-split patches with the best available larger model
   on Ada (e.g. `Qwen/Qwen2-VL-7B-Instruct`, explain mode, `evidence_only` input).
2. Keep only those where the class matches gold and `score_explanations` faithfulness = 100% with
   ≥ 2 checkable claims. Save as `data/explain_targets/train_targets.jsonl` and report how many survived.
3. Add `--target explain` to `scripts/run_finetune_qwen.py` that trains Qwen2-VL-2B LoRA on
   `{"change_class", "justification", ...}` targets with `evidence_only` inputs, and evaluates in
   explain mode with the same scoring.
4. Add a Slurm script variant for this.

## Deliverables checklist
- `scripts/check_no_leak.py` prints 0 leaks.
- Dry run on 3 patches works for each of the 4 models.
- `results/explain/` contains per-model JSONs, `explanation_scores.json/.csv`, `panels/`,
  `explanation_report.pdf`, `explanation_report.html`, `summary_scores.png`.
- README section "Explanation mode" with the commands to run each step.
- Don't change existing result files; don't commit images or checkpoints.
