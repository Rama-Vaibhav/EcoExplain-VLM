"""Condition C VLM combo registry and JSONL dataset loader.

Combos = (vision encoder, projector, text decoder). This module does not
download models until you call ``eval_combo`` / notebook 05.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

CLASSES = ["stable", "decline", "increase"]


@dataclass(frozen=True)
class VLMCombo:
    combo_id: str
    family: str
    vision_encoder: str
    projector: str
    text_decoder: str
    hf_model_id: Optional[str]
    notes: str
    recommended: bool = False
    # Smaller HF stand-in that can finish on a 16 GB Mac. Same family, not the 7B paper weights.
    laptop_hf_model_id: Optional[str] = None


# Real-world families from the research table. hf_model_id is a runnable
# open-weight stand-in where one exists; None means architecture-only.
COMBOS: list[VLMCombo] = [
    VLMCombo(
        "llava15_clip_mlp_vicuna",
        "LLaVA-1.5",
        "CLIP-ViT-L/14",
        "2-layer MLP",
        "Vicuna / Llama-3",
        "llava-hf/llava-1.5-7b-hf",
        "Baseline: CLIP visual tokens → MLP → Vicuna. Solid semantic alignment.",
        recommended=True,
        laptop_hf_model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
    ),
    VLMCombo(
        "llava16_clip_mlp_vicuna",
        "LLaVA-1.6",
        "CLIP-ViT-L/14 (higher-res grid)",
        "2-layer MLP",
        "Vicuna / Mistral / Llama-3",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Same CLIP+MLP pattern, better high-res tiling for 512×512 T0/T1.",
    ),
    VLMCombo(
        "paligemma_siglip_linear_gemma",
        "PaliGemma",
        "SigLIP",
        "Linear / MLP",
        "Gemma",
        "google/paligemma-3b-mix-224",
        "Sigmoid contrastive vision; efficient at higher resolution than CLIP softmax.",
        recommended=True,
        laptop_hf_model_id="google/paligemma-3b-mix-224",
    ),
    VLMCombo(
        "blip2_clip_qformer_opt",
        "BLIP-2",
        "CLIP / ViT",
        "Q-Former (cross-attention queries)",
        "OPT / Flan-T5",
        "Salesforce/blip2-opt-2.7b",
        "Compresses many patch tokens into a small query set before the LLM.",
        laptop_hf_model_id="Salesforce/blip2-opt-2.7b",
    ),
    VLMCombo(
        "internvl_internvit_mlp_internlm",
        "InternVL",
        "InternViT",
        "MLP",
        "InternLM",
        "OpenGVLab/InternVL2-8B",
        "Hybrid high-res vision tower; strong on dense RS-like detail.",
    ),
    VLMCombo(
        "qwen_vl_custom_mlp_qwen",
        "Qwen2-VL",
        "Custom ViT (dynamic resolution)",
        "MLP / merger",
        "Qwen2",
        "Qwen/Qwen2-VL-7B-Instruct",
        "Native multimodal LLM; good multilingual + instruction following.",
        recommended=True,
        laptop_hf_model_id="Qwen/Qwen2-VL-2B-Instruct",
    ),
    VLMCombo(
        "eve_encoder_free",
        "Encoder-Free (EVE)",
        "None (patches into unified decoder)",
        "None",
        "Unified LLM decoder",
        None,
        "No separate vision encoder. Research-only; not a first EcoExplain run.",
    ),
]


def combo_table() -> list[dict[str, Any]]:
    return [
        {
            "combo_id": c.combo_id,
            "family": c.family,
            "vision_encoder": c.vision_encoder,
            "projector": c.projector,
            "text_decoder": c.text_decoder,
            "hf_model_id": c.hf_model_id,
            "laptop_hf_model_id": c.laptop_hf_model_id,
            "recommended": c.recommended,
            "notes": c.notes,
        }
        for c in COMBOS
    ]


def resolve_checkpoint(combo: VLMCombo, laptop: bool = False) -> VLMCombo:
    """Use the laptop-sized HF id when RAM/disk cannot hold 7B weights."""
    from dataclasses import replace

    if not laptop:
        return combo
    lid = combo.laptop_hf_model_id or combo.hf_model_id
    if lid == combo.hf_model_id:
        return combo
    return replace(combo, hf_model_id=lid)


def get_combo(combo_id: str) -> VLMCombo:
    for c in COMBOS:
        if c.combo_id == combo_id:
            return c
    known = [c.combo_id for c in COMBOS]
    raise KeyError(f"Unknown combo {combo_id!r}. Choose one of {known}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_split_ids(splits_dir: Path, split: str) -> list[str]:
    csv_path = splits_dir / f"{split}_patches.csv"
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df["patch_id"].astype(str).tolist()


MINIMAL_VISION_PROMPT = (
    "Two forest RGB patches: image 1 = T0 (Jan–Feb 2022), image 2 = T1 (Jan–Feb 2023). "
    "No spectral tables — judge change from the images only."
)

# Tokens that name the gold label or the analyst's class decision.
LABEL_LEAK_TOKENS = (
    "candidate_class",
    "vegetation_trend",
    "reference_class",
    "partial_mismatch",
)


def clean_evidence_text(record: dict[str, Any]) -> str:
    """Condition C measurements with the gold label and the inference line removed."""
    import re

    raw = str((record.get("text") or {}).get("condition_c_user_prompt") or "")
    kept: list[str] = []
    for line in raw.splitlines():
        low = line.lower().strip()
        if low.startswith("inference for"):
            continue
        if low.startswith("respond with class"):
            continue
        for tok in LABEL_LEAK_TOKENS:
            line = re.sub(rf"{tok}\s*=\s*\S+", "", line, flags=re.IGNORECASE)
            line = re.sub(rf"\b{tok}\b", "", line, flags=re.IGNORECASE)
        if line.strip():
            kept.append(line.rstrip())
    return "\n".join(kept).strip()


def text_has_label_leak(text: str, gold: Optional[str] = None) -> Optional[str]:
    """Return the leaked token, or None. Gold is matched only as '=label', not as a bare word."""
    import re

    low = (text or "").lower()
    for tok in LABEL_LEAK_TOKENS:
        if tok in low:
            return tok
    if gold and re.search(rf"=\s*{re.escape(gold)}\b", text or "", flags=re.IGNORECASE):
        return f"={gold}"
    return None


def encoder_text(record: dict[str, Any], *, text_mode: str = "full") -> str:
    """String that goes to the text tokenizer / LLM prefix."""
    if text_mode == "minimal":
        return MINIMAL_VISION_PROMPT
    if text_mode == "evidence_only":
        return clean_evidence_text(record)
    if text_mode != "full":
        raise ValueError(f"Unknown text_mode {text_mode!r}")
    text = (record.get("text") or {}).get("condition_c_user_prompt")
    if text:
        return str(text)
    # Fallback if someone still has v1 JSONL
    parts = [
        f"Task: {record.get('task')}",
        f"Candidate prior: {record.get('candidate_class')}",
        record.get("explanation") or "",
    ]
    return "\n".join(parts)


def target_text(record: dict[str, Any]) -> str:
    """Supervision string: class + grounded explanation."""
    cls = record.get("candidate_class") or "unknown"
    expl = record.get("explanation") or ""
    inf = ((record.get("file_analysis") or {}).get("inference") or {}).get("vegetation_trend")
    return json.dumps(
        {
            "change_class": cls,
            "vegetation_trend": inf,
            "disturbance_present": bool(record.get("triplets")),
            "explanation": expl,
        },
        ensure_ascii=False,
    )


def image_paths(record: dict[str, Any], root: Path) -> tuple[Path, Path]:
    imgs = record.get("images") or {}
    t0 = root / imgs["t0"]
    t1 = root / imgs["t1"]
    return t0, t1


def load_t0_t1(record: dict[str, Any], root: Path) -> tuple[Image.Image, Image.Image]:
    t0, t1 = image_paths(record, root)
    if not t0.exists() or not t1.exists():
        raise FileNotFoundError(f"Missing T0/T1 JPGs for {record.get('id')}: {t0} {t1}")
    return Image.open(t0).convert("RGB"), Image.open(t1).convert("RGB")


def records_for_split(jsonl_path: Path, splits_dir: Path, split: str) -> list[dict[str, Any]]:
    want = set(load_split_ids(splits_dir, split))
    return [r for r in load_jsonl(jsonl_path) if r["id"] in want]


def condition_c_sample(
    record: dict[str, Any], root: Path, *, text_mode: str = "full"
) -> dict[str, Any]:
    """One training/eval example for any combo."""
    t0, t1 = load_t0_t1(record, root)
    return {
        "id": record["id"],
        "images": [t0, t1],  # vision encoder input (two RGB chips)
        "input_text": encoder_text(record, text_mode=text_mode),
        "target_text": target_text(record),
        "label": record.get("candidate_class"),
        "label_id": CLASSES.index(record["candidate_class"]) if record.get("candidate_class") in CLASSES else -1,
    }
