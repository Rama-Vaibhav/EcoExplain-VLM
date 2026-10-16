"""Faithfulness, self-consistency, and copy-overlap scores for model justifications."""

from __future__ import annotations

import re
from typing import Any, Optional

from src.vlm_evidence import measured_deltas

# |delta| below the threshold counts as stable for that indicator.
DIRECTION_THRESHOLDS = {
    "NDVI": 0.02,
    "EVI": 0.02,
    "NDMI": 0.02,
    "NBR": 0.02,
    "green_cover": 0.05,
    "VV": 0.3,
    "VH": 0.3,
    "rainfall": 5.0,
    "soil_moisture": 0.01,
    "temperature": 0.3,
}

_INDICATORS = (
    ("NDVI", r"\bndvi\b"),
    ("EVI", r"\bevi\b"),
    ("NDMI", r"\bndmi\b"),
    ("NBR", r"\bnbr\b"),
    ("green_cover", r"\b(green cover|greenness|canopy)\b"),
    ("VV", r"\bvv\b"),
    ("VH", r"\b(vh|backscatter)\b"),
    ("rainfall", r"\b(rainfall|precipitation|drier|wetter)\b"),
    ("soil_moisture", r"\bsoil moisture\b"),
    ("temperature", r"\b(temperature|warmer|cooler)\b"),
)

_DIRECTIONS = (
    ("increase", r"\b(increase|increased|increasing|rise|rising|risen|higher|gain|gained|wetter|warmer)\b"),
    ("decrease", r"\b(decrease|decreased|decreasing|drop|dropped|decline|declined|declining|lower|loss|drier|cooler)\b"),
    ("stable", r"\b(stable|unchanged|little change|no change|similar)\b"),
)

_CLASS_DIRECTION = {"decline": "decrease", "increase": "increase", "stable": "stable"}


def direction_of(delta: Optional[float], indicator: str) -> Optional[str]:
    if delta is None:
        return None
    thresh = DIRECTION_THRESHOLDS[indicator]
    if abs(delta) < thresh:
        return "stable"
    return "increase" if delta > 0 else "decrease"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9Δ]+", (text or "").lower())


def rouge_l(hyp: str, ref: str) -> float:
    """Token ROUGE-L F1. Implemented here so scoring does not need an extra package."""
    a, b = _tokens(hyp), _tokens(ref)
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[m]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def distinct_2(texts: list[str]) -> float:
    bigrams: list[tuple[str, str]] = []
    for text in texts:
        toks = _tokens(text)
        bigrams.extend(zip(toks, toks[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


def near_duplicate_share(texts: list[str], cutoff: float = 0.9) -> float:
    if len(texts) < 2:
        return 0.0
    dup = 0
    for i, text in enumerate(texts):
        if any(i != j and rouge_l(text, other) > cutoff for j, other in enumerate(texts)):
            dup += 1
    return dup / len(texts)


def justification_text(parsed: dict[str, Any], raw: str = "") -> str:
    parts = [str(parsed.get("justification") or "")]
    for key in ("evidence_bullets", "supporting_evidence", "conflicting_evidence", "visual_change", "possible_driver"):
        val = parsed.get(key)
        if isinstance(val, list):
            parts.extend(str(x) for x in val)
        elif val:
            parts.append(str(val))
    text = " ".join(p for p in parts if p).strip()
    return text or (raw or "")


def _first_direction(snippet: str) -> Optional[str]:
    best_at = None
    best = None
    for label, pattern in _DIRECTIONS:
        match = re.search(pattern, snippet, flags=re.IGNORECASE)
        if match and (best_at is None or match.start() < best_at):
            best_at = match.start()
            best = label
    return best


def _nearest_direction(text: str, end: int, start: int) -> Optional[str]:
    """Prefer the first direction word after the indicator, then a word just before it."""
    right = _first_direction(text[end : end + 40])
    if right:
        return right
    left = text[max(0, start - 30) : start]
    # Search the left snippet from the indicator backward by reversing token scan.
    hits = []
    for label, pattern in _DIRECTIONS:
        for match in re.finditer(pattern, left, flags=re.IGNORECASE):
            hits.append((match.start(), label))
    if not hits:
        return None
    return max(hits)[1]


def extract_claims(text: str) -> list[dict[str, str]]:
    """Indicator plus a nearby direction word. One claim per indicator mention."""
    claims = []
    seen: set[tuple[str, int]] = set()
    for name, pattern in _INDICATORS:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            key = (name, match.start())
            if key in seen:
                continue
            word = match.group(0).lower()
            if word == "drier":
                direction = "decrease"
            elif word == "wetter":
                direction = "increase"
            elif word == "warmer":
                direction = "increase"
            elif word == "cooler":
                direction = "decrease"
            else:
                direction = _nearest_direction(text, match.end(), match.start())
            if direction is None:
                continue
            seen.add(key)
            claims.append({"indicator": name, "direction": direction})
    return claims


def score_justification(text: str, record: dict[str, Any], pred: Optional[str]) -> dict[str, Any]:
    claims = extract_claims(text)
    deltas = measured_deltas(record)
    checked = []
    for claim in claims:
        truth = direction_of(deltas.get(claim["indicator"]), claim["indicator"])
        if truth is None:
            continue
        ok = claim["direction"] == truth
        checked.append({**claim, "measured": truth, "correct": ok})
    n = len(checked)
    n_ok = sum(1 for c in checked if c["correct"])
    directions = [c["direction"] for c in claims]
    dominant = None
    if directions:
        # A tie has no dominant direction; picking one from a set made the score vary between runs.
        counts = {d: directions.count(d) for d in set(directions)}
        top = max(counts.values())
        leaders = [d for d, n in counts.items() if n == top]
        dominant = leaders[0] if len(leaders) == 1 else None
    expected = _CLASS_DIRECTION.get(pred or "")
    consistent = None if dominant is None or expected is None else dominant == expected
    return {
        "n_claims": n,
        "n_correct": n_ok,
        "faithfulness": (n_ok / n) if n else None,
        "has_wrong_claim": any(not c["correct"] for c in checked),
        "self_consistent": consistent,
        "claims": checked,
    }
