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
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    max_train_samples: Optional[int] = None  # None = all train patches
    max_text_chars: int = 2200
    grad_accum_steps: int = 4
    max_side: int = 384
    use_4bit: bool = True  # set False if bitsandbytes missing / CPU


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
    user_text = (sample["input_text"][: cfg.max_text_chars] + CLASS_ASK).strip()
    assistant_text = classification_target(sample["label"])
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
        lora_dropout=0.05,
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


def train_lora(
    root: Path,
    cfg: Optional[FinetuneConfig] = None,
) -> Path:
    import random

    import torch

    cfg = cfg or FinetuneConfig()
    adapter_path = root / cfg.adapter_dir
    adapter_path.mkdir(parents=True, exist_ok=True)

    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    train_records = records_for_split(jsonl, splits, "train")
    if cfg.max_train_samples:
        train_records = train_records[: cfg.max_train_samples]
    train_batch = collate(train_records, root, n=None)
    if not train_batch:
        raise RuntimeError("No training samples — check JPG paths under data/processed_images/")

    processor, model, device = load_qwen_for_train(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    global_step = 0
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
                global_step += 1
        print(f"epoch {epoch + 1}/{cfg.epochs} mean_loss={epoch_loss / max(len(order), 1):.4f}")

    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    (adapter_path / "finetune_config.json").write_text(
        json.dumps(
            {
                "base_model_id": cfg.base_model_id,
                "combo_id": FINETUNE_COMBO_ID,
                "epochs": cfg.epochs,
                "n_train": len(train_batch),
                "learning_rate": cfg.learning_rate,
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
) -> dict[str, Any]:
    cfg = cfg or FinetuneConfig()
    adapter_path = root / cfg.adapter_dir
    if not adapter_path.exists():
        raise FileNotFoundError(f"Train first or copy adapter to {adapter_path}")

    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    records = records_for_split(jsonl, splits, split)
    batch = collate(records, root, n=max_eval)

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

    report = {
        "combo_id": FINETUNE_COMBO_ID,
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
    out = root / "results" / f"vlm_combo_{FINETUNE_COMBO_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"{FINETUNE_COMBO_ID}: acc={format_accuracy_pct(acc)} "
        f"n={len(batch)} parsed={len(labeled)}"
    )
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
