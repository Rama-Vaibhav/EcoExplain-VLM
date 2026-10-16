"""Closed-set scoring for LLaVA 0.5B, PaliGemma 3B, and BLIP-2 OPT 2.7B.

Free generation on these three checkpoints collapsed: LLaVA and BLIP-2
predicted decline on every test patch, and BLIP-2 mostly echoed its prompt.
This module does not overwrite those zero-shot JSON files.

Each patch is shown as a square four-panel image (2022, 2023, red/green
change, 2023 with that change tinted). The prompt states measured NDVI, NBR,
and EVI deltas and the same cutoffs already used in dataset construction
(NDVI delta <= -0.02 decline, >= +0.02 increase, otherwise stable). It does
not include candidate_class. The model must assign probability to the three
class words; the word with the highest score is the prediction.

A three-number bias is fit on the dev split only, then frozen for test.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from src.vlm_combos import CLASSES, get_combo, load_t0_t1, records_for_split, resolve_checkpoint
from src.vlm_eval import format_accuracy_pct, load_hf_model, unload

CLOSEDSET_COMBOS = (
    "llava15_clip_mlp_vicuna",
    "paligemma_siglip_linear_gemma",
    "blip2_clip_qformer_opt",
)

# Cutoffs copied from src/extensive_descriptions.py::_infer_signals.
# Not tuned on the test split.
NDVI_DECLINE_AT = -0.02
NDVI_INCREASE_AT = 0.02

# Previous free-generation test accuracies (laptop checkpoints, n=55).
FREE_GEN_TEST_ACC = {
    "llava15_clip_mlp_vicuna": 0.32727272727272727,
    "paligemma_siglip_linear_gemma": 0.36363636363636365,
    "blip2_clip_qformer_opt": 0.32727272727272727,
    "qwen_vl_custom_mlp_qwen": 0.5636363636363636,
}


def answer_prompt(record: dict[str, Any]) -> str:
    """Short evidence prompt. Must not contain the gold candidate_class string."""
    fa = record.get("file_analysis") or {}
    delta = fa.get("index_delta") or {}
    ndvi = float(delta.get("NDVI") or 0.0)
    nbr = float(delta.get("NBR") or 0.0)
    evi = float(delta.get("EVI") or 0.0)
    return (
        "Top-left panel is forest in Jan-Feb 2022. Top-right is the same place in Jan-Feb 2023. "
        "Bottom-left is red where the canopy got darker and green where it got brighter. "
        "Bottom-right is 2023 with that color added. "
        f"NDVI delta is {ndvi:+.3f}. NBR delta is {nbr:+.3f}. EVI delta is {evi:+.3f}. "
        f"Use decline when NDVI delta is {NDVI_DECLINE_AT:.2f} or lower, "
        f"increase when NDVI delta is {NDVI_INCREASE_AT:+.2f} or higher, "
        "and stable otherwise. The vegetation change class is"
    )


def change_collage(im0: Image.Image, im1: Image.Image, tile: int = 224) -> Image.Image:
    """Square 2x2 image so a 224px encoder does not squash a wide side-by-side stitch."""
    import numpy as np

    a_im = im0.convert("RGB").resize((tile, tile))
    b_im = im1.convert("RGB").resize((tile, tile))
    a = np.asarray(a_im, dtype=np.float32)
    b = np.asarray(b_im, dtype=np.float32)
    delta = b.mean(axis=2) - a.mean(axis=2)
    scale = float(np.percentile(np.abs(delta), 95)) or 1.0
    norm = np.clip(delta / scale, -1.0, 1.0)
    diff = np.zeros_like(a)
    diff[..., 0] = np.clip(-norm, 0.0, 1.0) * 255.0
    diff[..., 1] = np.clip(norm, 0.0, 1.0) * 255.0
    diff[..., 2] = 32.0
    overlay = np.clip(b * 0.6 + diff * 0.4, 0, 255)
    canvas = Image.new("RGB", (tile * 2, tile * 2))
    canvas.paste(a_im, (0, 0))
    canvas.paste(b_im, (tile, 0))
    canvas.paste(Image.fromarray(diff.astype(np.uint8), mode="RGB"), (0, tile))
    canvas.paste(Image.fromarray(overlay.astype(np.uint8), mode="RGB"), (tile, tile))
    return canvas


def _family(combo_id: str) -> str:
    if combo_id.startswith("blip2"):
        return "blip2"
    if combo_id.startswith("paligemma"):
        return "paligemma"
    return "llava"


def _move_to_model(inputs: dict[str, Any], model) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved


def _prepare_inputs(processor, image: Image.Image, prompt: str, family: str) -> dict[str, Any]:
    if family == "blip2":
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    elif family == "paligemma":
        text = prompt if prompt.startswith("<image>") else "<image>" + prompt
        inputs = processor(images=image, text=text, return_tensors="pt")
    else:
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            templated = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=[image], text=templated, return_tensors="pt")
        except Exception:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
    return dict(inputs)


def _continuation_ids(processor, word: str) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Tokenizer produced no ids for {word!r}")
    return list(ids)


def _forward(model, batch: dict[str, Any]):
    import inspect

    try:
        return model(**batch)
    except TypeError:
        allowed = set(inspect.signature(model.forward).parameters)
        filtered = {k: v for k, v in batch.items() if k in allowed}
        return model(**filtered)


def _extend_batch(base: dict[str, Any], extra_ids: list[int]) -> dict[str, Any]:
    """Append answer tokens. Drop labels so PaliGemma does not run a loss on a shorter target."""
    import torch

    ids = base["input_ids"]
    seq_len = ids.shape[-1]
    n_extra = len(extra_ids)
    extra = torch.tensor([extra_ids], device=ids.device, dtype=ids.dtype)
    full: dict[str, Any] = {}
    for key, value in base.items():
        if key == "labels":
            continue
        if not hasattr(value, "shape") or getattr(value, "ndim", 0) != 2 or value.shape[-1] != seq_len:
            full[key] = value
            continue
        if key == "input_ids":
            full[key] = torch.cat([value, extra], dim=1)
        elif key == "position_ids":
            start = int(value[0, -1].item()) + 1
            pos = torch.arange(start, start + n_extra, device=value.device, dtype=value.dtype).unsqueeze(0)
            full[key] = torch.cat([value, pos], dim=1)
        elif key in {"token_type_ids", "mm_token_type_ids"}:
            # 1 = suffix text. 0 would mark the answer as another image token.
            ones = torch.ones(1, n_extra, device=value.device, dtype=value.dtype)
            full[key] = torch.cat([value, ones], dim=1)
        elif key == "attention_mask":
            ones = torch.ones(1, n_extra, device=value.device, dtype=value.dtype)
            full[key] = torch.cat([value, ones], dim=1)
        else:
            zeros = torch.zeros(1, n_extra, device=value.device, dtype=value.dtype)
            full[key] = torch.cat([value, zeros], dim=1)
    return full


def _mean_logprob(model, processor, base: dict[str, Any], word: str) -> float:
    import torch
    import torch.nn.functional as F

    extra = _continuation_ids(processor, word)
    base_len = base["input_ids"].shape[-1]
    full = _extend_batch(base, extra)
    with torch.inference_mode():
        logits = _forward(model, full).logits[0].float()
    total = 0.0
    for i, tok in enumerate(extra):
        pos = base_len - 1 + i
        total += F.log_softmax(logits[pos], dim=-1)[tok].item()
    return total / len(extra)


def score_classes(model, processor, image: Image.Image, prompt: str, family: str) -> dict[str, float]:
    base = _move_to_model(_prepare_inputs(processor, image, prompt, family), model)
    return {label: _mean_logprob(model, processor, base, label) for label in CLASSES}


def _predict(logp: dict[str, float], bias: dict[str, float]) -> str:
    return max(CLASSES, key=lambda label: logp[label] + bias.get(label, 0.0))


def _accuracy(rows: list[dict[str, Any]], bias: dict[str, float]) -> float:
    if not rows:
        return 0.0
    correct = sum(_predict(row["logp"], bias) == row["gold"] for row in rows)
    return correct / len(rows)


def fit_dev_bias(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    """Additive class bias. Grid is fit on dev rows only."""
    grid = (-1.0, -0.5, 0.0, 0.5, 1.0)
    best_bias = {label: 0.0 for label in CLASSES}
    best_acc = _accuracy(rows, best_bias)
    for triple in itertools.product(grid, repeat=3):
        bias = dict(zip(CLASSES, triple))
        acc = _accuracy(rows, bias)
        if acc > best_acc:
            best_acc = acc
            best_bias = bias
    return best_bias, best_acc


def _samples(records: list[dict[str, Any]], root: Path, n: Optional[int]) -> list[dict[str, Any]]:
    subset = records if n is None else records[:n]
    out = []
    for rec in subset:
        prompt = answer_prompt(rec)
        if "candidate_class" in prompt:
            raise RuntimeError(f"Prompt leaked candidate_class for {rec.get('id')}")
        try:
            im0, im1 = load_t0_t1(rec, root)
        except FileNotFoundError as exc:
            print("SKIP", rec.get("id"), exc)
            continue
        out.append(
            {
                "id": rec["id"],
                "gold": rec.get("candidate_class"),
                "image": change_collage(im0, im1),
                "prompt": prompt,
            }
        )
    return out


def _score_split(model, processor, samples: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    rows = []
    for i, sample in enumerate(samples, start=1):
        logp = score_classes(model, processor, sample["image"], sample["prompt"], family)
        rows.append({"id": sample["id"], "gold": sample["gold"], "logp": logp, "prompt": sample["prompt"]})
        if i % 10 == 0 or i == len(samples):
            print(f"  scored {i}/{len(samples)}", flush=True)
    return rows


def _pack(
    rows: list[dict[str, Any]],
    bias: dict[str, float],
) -> list[dict[str, Any]]:
    preds = []
    for row in rows:
        pred = _predict(row["logp"], bias)
        preds.append(
            {
                "id": row["id"],
                "gold": row["gold"],
                "pred": pred,
                "correct": pred == row["gold"],
                "logp": {k: round(v, 4) for k, v in row["logp"].items()},
            }
        )
    return preds


def eval_closedset_combo(
    root: Path,
    combo_id: str,
    *,
    laptop: bool = True,
    max_eval: int = 55,
) -> dict[str, Any]:
    combo = resolve_checkpoint(get_combo(combo_id), laptop=laptop)
    family = _family(combo.combo_id)
    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    dev_samples = _samples(records_for_split(jsonl, splits, "dev"), root, max_eval)
    test_samples = _samples(records_for_split(jsonl, splits, "test"), root, max_eval)
    t0 = time.time()
    processor, model, device = load_hf_model(combo)
    print(f"{combo.combo_id}: scoring dev ({len(dev_samples)}) then test ({len(test_samples)}) on {device}")
    try:
        dev_rows = _score_split(model, processor, dev_samples, family)
        test_rows = _score_split(model, processor, test_samples, family)
    finally:
        unload(model, processor)

    zero = {label: 0.0 for label in CLASSES}
    bias, dev_cal = fit_dev_bias(dev_rows)
    dev_raw = _accuracy(dev_rows, zero)
    test_raw = _accuracy(test_rows, zero)
    test_cal = _accuracy(test_rows, bias)
    report = {
        "combo_id": combo.combo_id,
        "hf_model_id": combo.hf_model_id,
        "laptop": laptop,
        "decode": "closed_set_logprob",
        "prompt": "four_panel_collage_plus_ndvi_rule",
        "ndvi_rule": {"decline_at_or_below": NDVI_DECLINE_AT, "increase_at_or_above": NDVI_INCREASE_AT},
        "label_in_prompt": False,
        "dev_bias": bias,
        "dev_accuracy_raw": dev_raw,
        "dev_accuracy_calibrated": dev_cal,
        "test_accuracy_raw": test_raw,
        "accuracy": test_cal,
        "test_accuracy_calibrated": test_cal,
        "free_generation_accuracy": FREE_GEN_TEST_ACC.get(combo.combo_id),
        "qwen_zero_shot_accuracy": FREE_GEN_TEST_ACC["qwen_vl_custom_mlp_qwen"],
        "n_dev": len(dev_rows),
        "n_eval": len(test_rows),
        "seconds": round(time.time() - t0, 2),
        "status": "ok",
        "note": (
            "Test accuracy is closed-set logprob after a dev-only class bias. "
            "candidate_class is not in the prompt. "
            "This does not replace the original free-generation JSON."
        ),
        "predictions": _pack(test_rows, bias),
    }
    out = root / "results" / f"vlm_closedset_{combo.combo_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"{combo.combo_id}: raw={format_accuracy_pct(test_raw)} "
        f"calibrated={format_accuracy_pct(test_cal)} "
        f"dev_raw={format_accuracy_pct(dev_raw)} free_gen={format_accuracy_pct(report['free_generation_accuracy'])}"
    )
    return report


def run_closedset(
    root: Path,
    combo_ids: Optional[list[str]] = None,
    *,
    laptop: bool = True,
    max_eval: int = 55,
) -> dict[str, Any]:
    ids = list(combo_ids or CLOSEDSET_COMBOS)
    reports = []
    for combo_id in ids:
        try:
            reports.append(eval_closedset_combo(root, combo_id, laptop=laptop, max_eval=max_eval))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"{combo_id}: FAILED {exc}")
            reports.append(
                {
                    "combo_id": combo_id,
                    "status": "failed",
                    "error": str(exc),
                    "accuracy": None,
                    "test_accuracy_raw": None,
                }
            )
    ranking = sorted(
        [r for r in reports if r.get("accuracy") is not None],
        key=lambda r: r["accuracy"],
        reverse=True,
    )
    summary = {
        "decode": "closed_set_logprob",
        "qwen_zero_shot_accuracy": FREE_GEN_TEST_ACC["qwen_vl_custom_mlp_qwen"],
        "qwen_zero_shot_accuracy_pct": format_accuracy_pct(FREE_GEN_TEST_ACC["qwen_vl_custom_mlp_qwen"]),
        "ranking": [
            {
                "combo_id": r["combo_id"],
                "hf_model_id": r.get("hf_model_id"),
                "test_accuracy_raw": r.get("test_accuracy_raw"),
                "test_accuracy_calibrated": r.get("accuracy"),
                "test_accuracy_calibrated_pct": format_accuracy_pct(r.get("accuracy")),
                "free_generation_accuracy": r.get("free_generation_accuracy"),
                "beats_qwen_56_4": bool(
                    r.get("accuracy") is not None and r["accuracy"] > FREE_GEN_TEST_ACC["qwen_vl_custom_mlp_qwen"]
                ),
            }
            for r in ranking
        ],
        "failures": [r for r in reports if r.get("status") != "ok"],
    }
    path = root / "results" / "vlm_closedset_comparison.json"
    path.write_text(json.dumps(summary, indent=2))
    print("wrote", path)
    for row in summary["ranking"]:
        flag = "above 56.4%" if row["beats_qwen_56_4"] else "not above 56.4%"
        print(
            f"  {row['combo_id']}: calibrated {row['test_accuracy_calibrated_pct']} "
            f"(raw {format_accuracy_pct(row['test_accuracy_raw'])}) {flag}"
        )
    return summary
