"""LoRA fine-tune Qwen2-VL on EcoExplain Condition C (train split only)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.vlm_combos import CLASSES, condition_c_sample, records_for_split
from src.vlm_eval import CLASS_ASK, collate, format_accuracy_pct, generate_one, parse_class, pick_device

FINETUNE_COMBO_ID = "qwen_vl_finetuned_ecoexplain"
DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_ADAPTER_DIR = "results/checkpoints/qwen2vl_lora_ecoexplain"


@dataclass
class FinetuneConfig:
    base_model_id: str = DEFAULT_BASE_MODEL
    adapter_dir: str = DEFAULT_ADAPTER_DIR
    epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    lora_r: int = 8
    lora_alpha: int = 16
    max_train_samples: Optional[int] = None  # None = all train patches
    max_text_chars: int = 2200
    grad_accum_steps: int = 4
    max_side: int = 384
    use_4bit: bool = True  # set False if bitsandbytes missing / CPU
    early_stopping: bool = True
    early_stopping_patience: int = 1  # stop after this many epochs without dev gain
    eval_dev_each_epoch: bool = True
    train_text_mode: str = "evidence_only"  # never "full" (that prompt contains candidate_class)
    target_mode: str = "class"  # or "explain"
    explain_targets: str = "data/explain_targets/train_targets.jsonl"


def classification_target(label: str) -> str:
    return json.dumps({"change_class": label}, ensure_ascii=False)


def _qwen_messages(images, user_text: str, assistant_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]


def _resize_images(images: list, max_side: int) -> list:
    out = []
    for im in images:
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        out.append(im)
    return out


def _build_supervised_batch(processor, sample: dict[str, Any], cfg: FinetuneConfig, device):
    import torch

    images = _resize_images(sample["images"], cfg.max_side)
    ask = sample.get("ask_suffix") or CLASS_ASK
    user_text = (sample["input_text"][: cfg.max_text_chars] + ask).strip()
    assistant_text = sample.get("assistant_text") or classification_target(sample["label"])
    messages = _qwen_messages(images, user_text, assistant_text)

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[full_text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    # Mask prompt tokens: only train on assistant answer tokens.
    prompt_only = processor.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = processor(text=[prompt_only], images=images, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    labels[:, :prompt_len] = -100
    inputs["labels"] = labels
    return inputs


def load_qwen_for_train(cfg: FinetuneConfig):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = pick_device()
    processor = AutoProcessor.from_pretrained(cfg.base_model_id, trust_remote_code=True)

    quant_kwargs = {}
    if cfg.use_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            cfg.use_4bit = False

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.base_model_id,
        torch_dtype=dtype if not quant_kwargs else None,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        **quant_kwargs,
    )
    if quant_kwargs:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    if device == "mps" and not quant_kwargs:
        model = model.to(device)
    model.train()
    return processor, model, device


def load_qwen_for_eval(cfg: FinetuneConfig, adapter_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = pick_device()
    processor = AutoProcessor.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.to(device) if device != "cuda" else model
    model.eval()
    return processor, model, device


def _split_accuracy(
    model,
    processor,
    root: Path,
    cfg: FinetuneConfig,
    split: str,
    max_eval: Optional[int] = None,
    text_mode: str = "evidence_only",
) -> tuple[float | None, int, int]:
    """Generation accuracy on a split (dev/test); does not write JSON."""
    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    records = records_for_split(jsonl, splits, split)
    batch = collate(records, root, n=max_eval, text_mode=text_mode)
    gen_kw = {"max_new_tokens": 48, "do_sample": False}
    model.eval()
    correct = 0
    parsed = 0
    for s in batch:
        try:
            text = generate_one(processor, model, s, gen_kw, max_side=cfg.max_side)
        except Exception:
            text = ""
        pred_cls = parse_class(text)
        if pred_cls is not None:
            parsed += 1
            if pred_cls == s["label"]:
                correct += 1
    model.train()
    acc = correct / parsed if parsed else None
    return acc, len(batch), parsed


def _attach_explain_targets(root: Path, cfg: FinetuneConfig, records: list[dict], batch: list[dict]) -> list[dict]:
    """Swap in model-written JSON targets. Drops patches that failed the faithfulness filter."""
    from src.vlm_eval import EXPLAIN_JSON_ASK
    from src.vlm_evidence import compact_evidence

    path = root / cfg.explain_targets
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it with scripts/build_explain_targets.py on Ada first."
        )
    targets = {
        json.loads(line)["id"]: json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    }
    by_id = {rec["id"]: rec for rec in records}
    kept = []
    for sample in batch:
        tgt = targets.get(sample["id"])
        rec = by_id.get(sample["id"])
        if not tgt or rec is None:
            continue
        sample["input_text"] = compact_evidence(rec)
        sample["assistant_text"] = tgt.get("target") or tgt.get("target_json")
        sample["ask_suffix"] = EXPLAIN_JSON_ASK
        if not sample["assistant_text"]:
            continue
        kept.append(sample)
    print(f"explain targets kept {len(kept)} / {len(batch)} train patches")
    if not kept:
        raise RuntimeError(f"No overlapping explain targets in {path}")
    return kept


def train_lora(
    root: Path,
    cfg: Optional[FinetuneConfig] = None,
) -> Path:
    import random
    import shutil

    import torch

    cfg = cfg or FinetuneConfig()
    adapter_path = root / cfg.adapter_dir
    best_path = adapter_path / "best"
    adapter_path.mkdir(parents=True, exist_ok=True)

    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    train_records = records_for_split(jsonl, splits, "train")
    if cfg.max_train_samples:
        train_records = train_records[: cfg.max_train_samples]
    if cfg.train_text_mode == "full" or cfg.target_mode not in {"class", "explain"}:
        raise ValueError("Training text_mode must be evidence_only or minimal, never full")
    train_batch = collate(train_records, root, n=None, text_mode=cfg.train_text_mode)
    if cfg.target_mode == "explain":
        train_batch = _attach_explain_targets(root, cfg, train_records, train_batch)
    if not train_batch:
        raise RuntimeError("No training samples — check JPG paths under data/processed_images/")

    processor, model, device = load_qwen_for_train(cfg)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    best_dev: float | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        order = list(range(len(train_batch)))
        random.shuffle(order)
        optim.zero_grad()
        for step_i, idx in enumerate(order):
            sample = train_batch[idx]
            inputs = _build_supervised_batch(processor, sample, cfg, device)
            loss = model(**inputs).loss / cfg.grad_accum_steps
            loss.backward()
            epoch_loss += float(loss.item()) * cfg.grad_accum_steps
            if (step_i + 1) % cfg.grad_accum_steps == 0:
                optim.step()
                optim.zero_grad()
        mean_loss = epoch_loss / max(len(order), 1)
        row: dict[str, Any] = {"epoch": epoch + 1, "train_mean_loss": round(mean_loss, 6)}

        if cfg.eval_dev_each_epoch:
            dev_acc, dev_n, dev_parsed = _split_accuracy(
                model, processor, root, cfg, "dev", text_mode=cfg.train_text_mode
            )
            row["dev_accuracy"] = dev_acc
            row["dev_n_eval"] = dev_n
            row["dev_n_parsed"] = dev_parsed
            print(
                f"epoch {epoch + 1}/{cfg.epochs} mean_loss={mean_loss:.4f} "
                f"dev_acc({cfg.train_text_mode})={format_accuracy_pct(dev_acc)} parsed={dev_parsed}/{dev_n}"
            )
            if dev_acc is not None and (best_dev is None or dev_acc > best_dev):
                best_dev = dev_acc
                best_epoch = epoch + 1
                stale_epochs = 0
                if best_path.exists():
                    shutil.rmtree(best_path)
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
                print(f"  -> new best dev checkpoint (epoch {best_epoch})")
            elif cfg.early_stopping:
                stale_epochs += 1
                if best_dev is not None and stale_epochs >= cfg.early_stopping_patience:
                    print(
                        f"early stop: no dev improvement for {stale_epochs} epoch(s) "
                        f"(best dev {format_accuracy_pct(best_dev)} @ epoch {best_epoch})"
                    )
                    history.append(row)
                    break
        else:
            print(f"epoch {epoch + 1}/{cfg.epochs} mean_loss={mean_loss:.4f}")
        history.append(row)

    if best_path.exists():
        for name in best_path.iterdir():
            dest = adapter_path / name.name
            if dest.is_dir():
                shutil.rmtree(dest, ignore_errors=True)
            else:
                dest.unlink(missing_ok=True)
        for name in best_path.iterdir():
            src, dest = name, adapter_path / name.name
            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
        print("restored best dev adapter to", adapter_path)
    else:
        model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)

    (adapter_path / "finetune_config.json").write_text(
        json.dumps(
            {
                "base_model_id": cfg.base_model_id,
                "combo_id": FINETUNE_COMBO_ID,
                "epochs_ran": len(history),
                "epochs_max": cfg.epochs,
                "best_dev_epoch": best_epoch,
                "best_dev_accuracy": best_dev,
                "n_train": len(train_batch),
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "early_stopping": cfg.early_stopping,
                "train_text_mode": cfg.train_text_mode,
                "target_mode": cfg.target_mode,
                "prompt_mode": cfg.train_text_mode,
                "history": history,
                "note": "Final test eval is reported separately; prefer dev for model selection.",
            },
            indent=2,
        )
    )
    print("saved adapter", adapter_path)
    return adapter_path


def eval_finetuned(
    root: Path,
    *,
    split: str = "test",
    max_eval: Optional[int] = 55,
    cfg: Optional[FinetuneConfig] = None,
    text_mode: str = "minimal",
    combo_id_suffix: str = "",
) -> dict[str, Any]:
    cfg = cfg or FinetuneConfig()
    if text_mode == "full":
        raise ValueError("Refusing text_mode='full' because that prompt contains candidate_class")
    adapter_path = root / cfg.adapter_dir
    if not adapter_path.exists():
        raise FileNotFoundError(f"Train first or copy adapter to {adapter_path}")

    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    records = records_for_split(jsonl, splits, split)
    batch = collate(records, root, n=max_eval, text_mode=text_mode)

    processor, model, device = load_qwen_for_eval(cfg, adapter_path)
    gen_kw = {"max_new_tokens": 48, "do_sample": False}

    t0 = time.time()
    preds = []
    for s in batch:
        try:
            text = generate_one(processor, model, s, gen_kw, max_side=cfg.max_side)
        except Exception as exc:
            text = f"[generate_error] {exc}"
        pred_cls = parse_class(text)
        preds.append(
            {
                "id": s["id"],
                "gold": s["label"],
                "pred_text": text[:2000],
                "pred": pred_cls,
                "correct": pred_cls == s["label"],
            }
        )

    labeled = [p for p in preds if p["pred"] is not None]
    acc = sum(p["correct"] for p in labeled) / len(labeled) if labeled else None
    golds = [p["gold"] for p in preds]
    maj = max(set(golds), key=golds.count) if golds else None
    majority_acc = golds.count(maj) / len(golds) if golds and maj else None

    combo_id = FINETUNE_COMBO_ID + combo_id_suffix
    report = {
        "combo_id": combo_id,
        "eval_split": split,
        "prompt_mode": text_mode,
        "family": "Qwen2-VL (LoRA fine-tuned)",
        "vision_encoder": "Custom ViT (dynamic resolution)",
        "projector": "MLP / merger + LoRA",
        "text_decoder": "Qwen2",
        "hf_model_id": cfg.base_model_id,
        "adapter_dir": str(cfg.adapter_dir),
        "laptop": False,
        "finetuned": True,
        "generation": gen_kw,
        "status": "ok",
        "error": None,
        "n_eval": len(batch),
        "n_parsed_class": len(labeled),
        "accuracy": acc,
        "majority_class": maj,
        "majority_accuracy": majority_acc,
        "seconds": round(time.time() - t0, 2),
        "predictions": preds,
    }
    out = root / "results" / f"vlm_combo_{combo_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"{combo_id}: acc={format_accuracy_pct(acc)} prompt={text_mode} "
        f"n={len(batch)} parsed={len(labeled)}"
    )
    return report


def eval_explain_finetuned(
    root: Path,
    *,
    max_eval: Optional[int] = 55,
    cfg: Optional[FinetuneConfig] = None,
) -> dict[str, Any]:
    """Explain-mode test eval. Writes results/explain/ and does not touch results/vlm_combo_*.json."""
    from src.explain_metrics import justification_text, score_justification
    from src.vlm_combos import load_t0_t1, records_for_split, text_has_label_leak
    from src.vlm_eval import generate_one, parse_explanation
    from src.vlm_evidence import compact_evidence

    cfg = cfg or FinetuneConfig()
    adapter_path = root / cfg.adapter_dir
    if not adapter_path.exists():
        raise FileNotFoundError(f"Train first or copy adapter to {adapter_path}")
    records = records_for_split(root / "dataset.jsonl", root / "data" / "splits", "test")
    if max_eval is not None:
        records = records[:max_eval]
    processor, model, device = load_qwen_for_eval(cfg, adapter_path)
    gen_kw = {"max_new_tokens": 256, "do_sample": False}
    rows = []
    t0 = time.time()
    try:
        for rec in records:
            evidence = compact_evidence(rec)
            leak = text_has_label_leak(evidence, rec.get("candidate_class"))
            if leak:
                raise AssertionError(f"{rec.get('id')} explain prompt contains {leak}")
            sample = {
                "id": rec["id"],
                "images": list(load_t0_t1(rec, root)),
                "input_text": evidence,
                "label": rec.get("candidate_class"),
            }
            try:
                raw = generate_one(
                    processor, model, sample, gen_kw, max_side=cfg.max_side, answer_format="json"
                )
            except Exception as exc:
                raw = f"[generate_error] {exc}"
            parsed = parse_explanation(raw)
            pred = parsed.get("change_class") or parse_class(raw)
            stats = score_justification(justification_text(parsed, raw), rec, pred)
            rows.append(
                {
                    "id": rec["id"],
                    "gold": rec.get("candidate_class"),
                    "pred": pred,
                    "correct": pred == rec.get("candidate_class"),
                    "raw_text": raw[:4000],
                    "parsed": parsed,
                    "prompt_mode": "explain",
                    "answer_format": "json",
                    "faithfulness": stats["faithfulness"],
                    "n_claims": stats["n_claims"],
                }
            )
    finally:
        unload_ok = True
        try:
            from src.vlm_eval import unload

            unload(model, processor)
        except Exception:
            unload_ok = False
    labeled = [r for r in rows if r["pred"] is not None]
    acc = sum(r["correct"] for r in labeled) / len(labeled) if labeled else None
    report = {
        "combo_id": FINETUNE_COMBO_ID,
        "hf_model_id": cfg.base_model_id,
        "adapter_dir": cfg.adapter_dir,
        "laptop": False,
        "finetuned": True,
        "mode": "explain",
        "prompt_mode": "explain",
        "answer_format": "json",
        "generation": gen_kw,
        "n_eval": len(rows),
        "n_parsed_class": len(labeled),
        "accuracy": acc,
        "seconds": round(time.time() - t0, 2),
        "predictions": rows,
        "note": "Scored further by scripts/score_explanations.py. Does not replace class-only result JSON.",
    }
    out = root / "results" / "explain" / f"explain_{FINETUNE_COMBO_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out} accuracy={format_accuracy_pct(acc)} unload={unload_ok}")
    return report


def merge_with_baselines(root: Path) -> dict[str, Any]:
    from src.vlm_eval import compare_reports

    combo_ids = [
        "llava15_clip_mlp_vicuna",
        "paligemma_siglip_linear_gemma",
        "qwen_vl_custom_mlp_qwen",
        "blip2_clip_qformer_opt",
        FINETUNE_COMBO_ID,
    ]
    reports = []
    for cid in combo_ids:
        path = root / "results" / f"vlm_combo_{cid}.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
    comparison = compare_reports(reports)
    comparison["includes_finetuned"] = any(r.get("combo_id") == FINETUNE_COMBO_ID for r in reports)
    (root / "results" / "vlm_combo_comparison_with_finetune.json").write_text(
        json.dumps(comparison, indent=2)
    )
    return comparison
