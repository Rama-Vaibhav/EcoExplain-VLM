"""Local LLM generation of VLM ground-truth explanations via Ollama."""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

DEFAULT_MODEL = "llama3"
FALLBACK_MODEL = "phi3"

SYSTEM_PROMPT = (
    "You are a remote-sensing analyst writing ground-truth captions for a "
    "Vision-Language Model benchmark (EcoExplain-VLM). "
    "Write 2–3 factual sentences that explain a forest disturbance. "
    "Always ground the claim in the bounding box [x_min, y_min, x_max, y_max] "
    "on the 512×512 VLM image, disturbance type (Fire or Deforestation), and "
    "severity (Low, Moderate, High). "
    "If supporting evidence is provided (Sentinel-1 VV/VH, CHIRPS rainfall, "
    "ERA5-Land temperature or soil-moisture anomalies, DEM slope/elevation), "
    "you may mention it only as corroboration — do not invent values. "
    "Do not mention place names, lat/lon, or that you are an AI. Stay concise."
)


def _format_triplets(triplets: Sequence[Sequence[Any]]) -> str:
    lines = []
    for i, item in enumerate(triplets, start=1):
        if len(item) < 3:
            raise ValueError(f"Triplet {i} must be [bbox, type, severity], got {item!r}")
        bbox, dtype, severity = item[0], item[1], item[2]
        lines.append(
            f"{i}. bounding_box={list(bbox)}, disturbance_type={dtype}, severity={severity}"
        )
    return "\n".join(lines) if lines else "(no disturbance regions detected)"


def _template_explanation(triplets: Sequence[Sequence[Any]]) -> str:
    """Deterministic caption used when Ollama is unavailable."""
    if not triplets:
        return (
            "No spatially coherent drop in ΔNDVI or ΔNBR exceeded the Otsu threshold "
            "in this chip. The before/after optical pair is consistent with stable "
            "canopy cover over the observation window."
        )
    parts = []
    for bbox, dtype, severity in triplets:
        x0, y0, x1, y1 = bbox
        if str(dtype).lower().startswith("fire"):
            parts.append(
                f"A {severity.lower()}-severity burn scar is localized at bounding box "
                f"[{x0}, {y0}, {x1}, {y1}], where ΔNBR (and typically ΔNDVI) show a "
                "significant negative drop isolated by Otsu thresholding."
            )
        else:
            parts.append(
                f"A {severity.lower()}-severity vegetation loss consistent with deforestation "
                f"is localized at bounding box [{x0}, {y0}, {x1}, {y1}], supported by a "
                "significant negative ΔNDVI drop after Otsu segmentation."
            )
    extra = (
        " These boxes are expressed in the padded 512×512 VLM image coordinates "
        "and should be treated as the spatial ground truth for this sample."
    )
    text = " ".join(parts)
    if len(parts) == 1:
        return text + extra
    return text + extra


def generate_vlm_explanation(
    triplets: Sequence[Sequence[Any]],
    *,
    model: str = DEFAULT_MODEL,
    host: Optional[str] = None,
    timeout: int = 120,
    context: Optional[dict] = None,
) -> dict:
    """Prompt a local Ollama model to write a 2–3 sentence disturbance explanation.

    Parameters
    ----------
    triplets :
        Iterable of ``[bbox, disturbance_type, severity]``.
    model :
        Ollama model name. Defaults to ``llama3``; ``phi3`` is tried if pull/run fails.
    host :
        Optional Ollama host URL (e.g. ``http://127.0.0.1:11434``).
    timeout :
        Client timeout in seconds.

    Returns
    -------
    dict
        ``explanation``, ``source`` (``ollama`` or ``fallback``), ``model``,
        and optional ``error``.
    """
    ctx = ""
    if context:
        ctx = "\nSupporting multi-sensor evidence (do not invent extra numbers):\n"
        ctx += json.dumps(context, default=str)
    user_prompt = (
        "Using the following auto-extracted triplets, write a 2–3 sentence factual "
        "explanation of the forest disturbance suitable as VLM ground-truth.\n\n"
        f"{_format_triplets(triplets)}\n{ctx}\n"
    )

    try:
        import httpx
        import ollama
    except ImportError as exc:
        return {
            "explanation": _template_explanation(triplets),
            "source": "fallback",
            "model": None,
            "error": f"ollama Python package is not installed ({exc}).",
        }

    client_kwargs = {"timeout": httpx.Timeout(timeout, connect=3.0)}
    if host:
        client_kwargs["host"] = host
    client = ollama.Client(**client_kwargs)

    models_to_try = [model]
    if model != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for name in models_to_try:
        try:
            response = client.chat(
                model=name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.7, "num_predict": 220},
            )
            content = (response.get("message") or {}).get("content", "").strip()
            if not content:
                last_error = f"Empty response from {name}"
                continue
            return {
                "explanation": content,
                "source": "ollama",
                "model": name,
                "error": None,
            }
        except Exception as exc:  # ConnectionError, ResponseError, timeout, missing model
            last_error = str(exc)
            continue

    return {
        "explanation": _template_explanation(triplets),
        "source": "fallback",
        "model": None,
        "error": (
            "Ollama is not running or no local model could be reached "
            f"(tried {models_to_try}). Last error: {last_error}. "
            "Start the daemon with `ollama serve` and pull `llama3` or `phi3`."
        ),
    }


def triplets_to_json(triplets: Sequence[Sequence[Any]]) -> str:
    """Serialize triplets for JSONL records."""
    records = []
    for item in triplets:
        bbox, dtype, severity = item[0], item[1], item[2]
        records.append({"bbox": list(bbox), "disturbance_type": dtype, "severity": severity})
    return json.dumps(records)
