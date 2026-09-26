"""Run Condition C combos on the same split and write comparison reports."""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from src.vlm_combos import (
    CLASSES,
    COMBOS,
    VLMCombo,
    condition_c_sample,
    get_combo,
    records_for_split,
    resolve_checkpoint,
)

# Same generation settings for a fair comparison (override per combo if needed).
DEFAULT_GEN = {
    "max_new_tokens": 192,
    "do_sample": False,
    "temperature": 1.0,
}

COMBO_GEN: dict[str, dict[str, Any]] = {
    "llava15_clip_mlp_vicuna": {**DEFAULT_GEN, "max_new_tokens": 192},
    "llava16_clip_mlp_vicuna": {**DEFAULT_GEN, "max_new_tokens": 192},
    "paligemma_siglip_linear_gemma": {**DEFAULT_GEN, "max_new_tokens": 96},
    "blip2_clip_qformer_opt": {**DEFAULT_GEN, "max_new_tokens": 64},
    "qwen_vl_custom_mlp_qwen": {**DEFAULT_GEN, "max_new_tokens": 192},
    "internvl_internvit_mlp_internlm": {**DEFAULT_GEN, "max_new_tokens": 192},
}


CLASS_ASK = (
    "\n\nReply with exactly one word after the images: decline or increase or stable. "
    "Do not copy the evidence text."
)

# BLIP-2 is single-image captioning; long Condition C text + dual-image chat templates yield empty output.
BLIP2_PROMPT = (
    "Side by side: left forest patch T0, right forest patch T1. "
    "One word only — decline, increase, or stable:"
)

# PaliGemma mix-224 often refuses long analytical prompts; keep text short.
PALIGEMMA_PROMPT_SUFFIX = (
    "Two forest RGB chips: first image T0, second T1. "
    "Vegetation change class — answer with exactly one word: stable, decline, or increase."
)


def format_accuracy_pct(acc: Optional[float]) -> str:
    """Human-readable accuracy for notebooks (0.5 -> '50.0%')."""
    if acc is None:
        return "—"
    return f"{acc * 100:.1f}%"


def normalize_class_token(word: str) -> Optional[str]:
    w = (word or "").lower().strip()
    if w in CLASSES:
        return w
    if w in {"decrease", "decreasing", "decreased", "loss", "negative", "declining"}:
        return "decline"
    if w in {"increase", "increasing", "increased", "gain", "growth", "positive"}:
        return "increase"
    if w in {"stable", "unchanged", "no_change", "no-change", "same", "neutral"}:
        return "stable"
    return None


def parse_class(text: str) -> Optional[str]:
    """Extract a class from *model* text only. Ignore the instruction template."""
    raw = (text or "").strip()
    t = raw.lower()
    # Instruction in the prompt looks like: "change_class": "stable"|"decline"|...
    # That must not count as a prediction.
    hits = list(
        re.finditer(r'"change_class"\s*:\s*"(stable|decline|increase)"(?!\s*\|)', t)
    )
    if hits:
        return hits[-1].group(1)
    # Single-word answers (common after "reply with one word").
    first = re.split(r"[\s,.;:!?\n]+", t.strip(), maxsplit=1)[0]
    one = normalize_class_token(first)
    if one and len(t) < 80:
        return one
    # Synonyms then canonical labels in longer text.
    for pat in (
        r"\b(decrease|decreasing|decline|declining)\b",
        r"\b(increase|increasing|growth|gain)\b",
        r"\b(stable|unchanged|neutral)\b",
    ):
        m = re.search(pat, t)
        if m:
            return normalize_class_token(m.group(1))
    hits = list(re.finditer(r"\b(stable|decline|increase)\b", t))
    if hits:
        return hits[-1].group(1)
    return None


def environment_report() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.executable}
    try:
        import transformers

        info["transformers"] = transformers.__version__
        info["transformers_file"] = transformers.__file__
        info["has_vision2seq"] = hasattr(transformers, "AutoModelForVision2Seq")
        info["has_image_text"] = hasattr(transformers, "AutoModelForImageTextToText")
    except Exception as exc:
        info["transformers_error"] = str(exc)
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = pick_device()
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception as exc:
        info["torchvision"] = None
        info["torchvision_error"] = str(exc)
    return info


def _import_vlm_model_class():
    """transformers 4.x: AutoModelForVision2Seq; 5.x: AutoModelForImageTextToText."""
    import transformers

    for name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls, name
    raise ImportError(
        "This Python environment cannot load VLMs. "
        f"python={sys.executable}. "
        "Select the ecoexplain .venv kernel (Python 3.9), not Anaconda. "
        "Need transformers>=4.45 with AutoModelForImageTextToText or AutoModelForVision2Seq."
    )


def hf_hub_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"


def purge_hf_repo(repo_id: str) -> None:
    """Delete one model's local Hub cache so the disk is not kept full of 7B shards."""
    import shutil

    d = hf_hub_dir(repo_id)
    locks = Path.home() / ".cache" / "huggingface" / "hub" / ".locks" / f"models--{repo_id.replace('/', '--')}"
    for p in (d, locks):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"purged local cache {p}")


def load_hf_model(combo: VLMCombo):
    import torch
    from transformers import AutoProcessor

    if not combo.hf_model_id:
        raise RuntimeError(f"{combo.combo_id} has no Hugging Face checkpoint")

    ModelCls, cls_name = _import_vlm_model_class()
    device = pick_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    processor = AutoProcessor.from_pretrained(combo.hf_model_id, trust_remote_code=True)
    last_err = None
    model = None
    try:
        model = ModelCls.from_pretrained(
            combo.hf_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        last_err = exc
        import transformers

        for alt in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "LlavaOnevisionForConditionalGeneration",
            "PaliGemmaForConditionalGeneration",
            "Blip2ForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ):
            cls = getattr(transformers, alt, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(
                    combo.hf_model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                cls_name = alt
                last_err = None
                break
            except Exception as exc2:
                last_err = exc2
    if model is None:
        raise RuntimeError(f"Could not load {combo.hf_model_id} via {cls_name}: {last_err}")

    model = model.to(device)
    model.eval()
    print(f"loaded {combo.hf_model_id} as {cls_name} on {device}")
    return processor, model, device


def collate(records, root: Path, n: Optional[int] = None) -> list[dict[str, Any]]:
    out = []
    subset = records if n is None else records[:n]
    for rec in subset:
        try:
            out.append(condition_c_sample(rec, root))
        except FileNotFoundError as exc:
            print("SKIP", rec.get("id"), exc)
    return out


def stitch_t0_t1(images: list) -> "Image.Image":
    from PIL import Image

    im0, im1 = images[0], images[1]
    w = max(im0.size[0], im1.size[0])
    h = max(im0.size[1], im1.size[1])
    im0 = im0.resize((w, h))
    im1 = im1.resize((w, h))
    out = Image.new("RGB", (w * 2, h))
    out.paste(im0, (0, 0))
    out.paste(im1, (w, 0))
    return out


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_one(
    processor,
    model,
    sample: dict[str, Any],
    gen_kw: dict[str, Any],
    *,
    max_side: Optional[int] = None,
) -> str:
    import torch

    prompt_text = (sample["input_text"][:1800] + CLASS_ASK).strip()
    images = sample["images"]
    if max_side:
        resized = []
        for im in images:
            w, h = im.size
            scale = min(1.0, max_side / max(w, h))
            resized.append(im.resize((max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else im)
        images = resized
    device = next(model.parameters()).device
    proc_name = type(processor).__name__

    if "Blip2" in proc_name:
        stitched = stitch_t0_t1(images)
        if max_side:
            w, h = stitched.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1:
                stitched = stitched.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        inputs = processor(images=stitched, text=BLIP2_PROMPT, return_tensors="pt")
    elif "PaliGemma" in proc_name:
        # Processor requires one <image> token per image, at the start of text.
        # Long Condition C paragraphs trigger "not trained to answer" refusals.
        brief = (sample["input_text"] or "").split("\n")
        hint = next((ln for ln in brief if "NDVI" in ln or "candidate_class" in ln), "")
        hint = hint[:400].strip()
        body = PALIGEMMA_PROMPT_SUFFIX
        if hint:
            body = hint + "\n" + body
        prompt_text = ("<image>" * len(images)) + body
        try:
            inputs = processor(images=images, text=prompt_text, return_tensors="pt")
        except Exception:
            inputs = processor(text=prompt_text, images=images[0], return_tensors="pt")
    else:
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=images, text=prompt, return_tensors="pt")
        except Exception:
            try:
                inputs = processor(images=images, text=prompt_text, return_tensors="pt")
            except Exception:
                inputs = processor(images=images[0], text=prompt_text, return_tensors="pt")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen_args = {k: v for k, v in gen_kw.items() if k != "temperature" or gen_kw.get("do_sample")}
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, **gen_args)
    new_tokens = out[0, input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        text = processor.decode(out[0], skip_special_tokens=True).strip()
    return text


def unload(model, processor) -> None:
    del model, processor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def eval_combo(
    combo: VLMCombo,
    batch: list[dict[str, Any]],
    results_dir: Path,
    dry_run: bool,
    *,
    laptop: bool = False,
    purge_after: bool = False,
) -> dict[str, Any]:
    combo = resolve_checkpoint(combo, laptop=laptop)
    gen_kw = dict(COMBO_GEN.get(combo.combo_id, DEFAULT_GEN))
    if laptop:
        gen_kw["max_new_tokens"] = min(int(gen_kw.get("max_new_tokens", 64)), 48)
    t0 = time.time()
    preds = []
    error = None
    status = "ok"

    if dry_run:
        for s in batch:
            preds.append(
                {
                    "id": s["id"],
                    "gold": s["label"],
                    "pred_text": "[dry-run]",
                    "pred": None,
                    "correct": False,
                }
            )
        status = "dry_run"
    elif combo.hf_model_id is None:
        status = "skipped_no_checkpoint"
        error = "No public HF checkpoint"
    else:
        processor = model = None
        try:
            processor, model, device = load_hf_model(combo)
            for s in batch:
                try:
                    text = generate_one(
                        processor, model, s, gen_kw, max_side=336 if laptop else None
                    )
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
        except Exception as exc:
            status = "load_failed"
            error = str(exc)
            for s in batch:
                preds.append(
                    {
                        "id": s["id"],
                        "gold": s["label"],
                        "pred_text": f"[load_failed] {exc}",
                        "pred": None,
                        "correct": False,
                    }
                )
        finally:
            if model is not None:
                unload(model, processor)
            if purge_after and combo.hf_model_id:
                purge_hf_repo(combo.hf_model_id)

    labeled = [p for p in preds if p["pred"] is not None]
    golds = [p["gold"] for p in preds]
    if golds:
        maj = max(set(golds), key=golds.count)
        majority_acc = golds.count(maj) / len(golds)
    else:
        maj, majority_acc = None, None
    if status == "ok" and labeled:
        acc = sum(p["correct"] for p in labeled) / len(labeled)
    elif status == "ok":
        acc = None  # model ran but no parseable class (do not show 0.0)
    else:
        acc = None
    report = {
        "combo_id": combo.combo_id,
        "family": combo.family,
        "vision_encoder": combo.vision_encoder,
        "projector": combo.projector,
        "text_decoder": combo.text_decoder,
        "hf_model_id": combo.hf_model_id,
        "laptop": laptop,
        "generation": gen_kw,
        "status": status,
        "error": error,
        "n_eval": len(batch),
        "n_parsed_class": len(labeled),
        "accuracy": acc,
        "majority_class": maj,
        "majority_accuracy": majority_acc,
        "seconds": round(time.time() - t0, 2),
        "predictions": preds,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"vlm_combo_{combo.combo_id}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"{combo.combo_id}: status={status} acc={format_accuracy_pct(acc)} "
        f"n={len(batch)} parsed={len(labeled)} s={report['seconds']}"
    )
    return report


def compare_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    runnable = [r for r in reports if r.get("status") == "ok" and r.get("accuracy") is not None]
    best = max(runnable, key=lambda r: r["accuracy"]) if runnable else None
    rows = [
        {
            "combo_id": r["combo_id"],
            "family": r["family"],
            "vision": r["vision_encoder"],
            "projector": r["projector"],
            "llm": r["text_decoder"],
            "max_new_tokens": r.get("generation", {}).get("max_new_tokens"),
            "do_sample": r.get("generation", {}).get("do_sample"),
            "status": r["status"],
            "accuracy": r.get("accuracy"),
            "accuracy_pct": format_accuracy_pct(r.get("accuracy")),
            "n_eval": r.get("n_eval"),
            "seconds": r.get("seconds"),
            "error": (r.get("error") or "")[:180],
        }
        for r in reports
    ]
    return {
        "n_combos": len(reports),
        "n_successful": len(runnable),
        "best_combo_id": None if best is None else best["combo_id"],
        "best_accuracy": None if best is None else best["accuracy"],
        "best_accuracy_pct": format_accuracy_pct(None if best is None else best["accuracy"]),
        "best_generation": None if best is None else best.get("generation"),
        "ranking": sorted(
            [r for r in rows if r["accuracy"] is not None],
            key=lambda r: r["accuracy"],
            reverse=True,
        ),
        "table": rows,
    }


def run_all(
    root: Path,
    *,
    split: str = "test",
    max_eval: int = 8,
    dry_run: bool = False,
    combo_ids: Optional[list[str]] = None,
    laptop: bool = False,
    purge_after: bool = False,
) -> dict[str, Any]:
    jsonl = root / "dataset.jsonl"
    splits = root / "data" / "splits"
    results = root / "results"
    records = records_for_split(jsonl, splits, split)
    batch = collate(records, root, n=max_eval)
    ids = combo_ids or [c.combo_id for c in COMBOS if c.hf_model_id]
    reports = []
    for cid in ids:
        combo = get_combo(cid)
        reports.append(
            eval_combo(
                combo,
                batch,
                results,
                dry_run=dry_run,
                laptop=laptop,
                purge_after=purge_after,
            )
        )
    comparison = compare_reports(reports)
    comparison["split"] = split
    comparison["max_eval"] = max_eval
    comparison["laptop"] = laptop
    comparison["same_patch_ids"] = [s["id"] for s in batch]
    (results / "vlm_combo_comparison.json").write_text(json.dumps(comparison, indent=2))
    return comparison
