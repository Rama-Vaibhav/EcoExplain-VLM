#!/usr/bin/env python3
"""C0.5 claim-level audit of the directional-claim checker (sir's Major 4.2-4.4, §3.5 denominators).

Re-runs the checker of src/explain_metrics.py claim by claim and saves every scored claim with
where it came from, so that all claim percentages in the paper trace to one documented file.
It also counts what the checker cannot see: unscored indicator mentions, truncated outputs,
claims that come from copied rule lines, negation and hedging near a claim.

    .venv/bin/python scripts/review/claim_audit.py
Outputs: results/review/claims.csv, results/review/claim_audit.json
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, rows, write_csv, write_json  # noqa: E402
from src.explain_metrics import (  # noqa: E402
    _INDICATORS, _nearest_direction, direction_of, justification_text, score_justification,
)
from src.vlm_evidence import measured_deltas  # noqa: E402

RUNS = {
    "llava_freeform": "explain/explain_llava15_clip_mlp_vicuna.json",
    "qwen_struct": "explain/explain_structured_qwen_vl_custom_mlp_qwen.json",
    "qwen_struct_512": "explain/explain_structured_qwen_vl_custom_mlp_qwen_tok512.json",
    "llava_struct": "explain/explain_structured_llava15_clip_mlp_vicuna.json",
    "qwen_expl_zs_336": "explain/explain_qwen_vl_custom_mlp_qwen.json",
    "qwen_expl_zs_384": "explain/explain_qwen_vl_custom_mlp_qwen_384.json",
    "expl_lora": "explain/explain_qwen_vl_finetuned_ecoexplain.json",
}
NEG = re.compile(r"\b(not|no|never|without|neither|nor|n't)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|possibly|probably|likely|slight|slightly|small|minor|suggest\w*)\b", re.I)
# Verbatim copy of the structured prompt's CLASS_RULE at the claim position, e.g. "NDVI Δ ≤ −0.02 → decline".
# The rule is derived from the card ΔNDVI with the same ±0.02 threshold the checker uses for NDVI,
# so a correctly copied rule line is "consistent" by construction.
RULE_COPY = re.compile(r"NDVI\s*Δ\s*(≤|≥|between)[^→]{0,30}→\s*(decline|increase|stable)")
SPECIAL = {"drier": "decrease", "wetter": "increase", "warmer": "increase", "cooler": "decrease"}


def claims_with_context(text: str) -> list[dict]:
    """Same claim extraction as src/explain_metrics.extract_claims, keeping the text around each claim."""
    out, seen = [], set()
    for name, pattern in _INDICATORS:
        for m in re.finditer(pattern, text or "", flags=re.I):
            key = (name, m.start())
            if key in seen:
                continue
            word = m.group(0).lower()
            direction = SPECIAL.get(word) or _nearest_direction(text, m.end(), m.start())
            if direction is None:
                out.append({"indicator": name, "direction": None, "start": m.start()})
                continue
            seen.add(key)
            out.append({"indicator": name, "direction": direction, "start": m.start(),
                        "window": text[max(0, m.start() - 30): m.end() + 40]})
    return out


def line_of(text: str, pos: int) -> str:
    a = text.rfind("\n", 0, pos) + 1
    b = text.find("\n", pos)
    return text[a: b if b != -1 else len(text)]


def main() -> None:
    import json

    recs = rows()
    table, summary = [], {}
    for run, rel in RUNS.items():
        path = RESULTS / rel
        if not path.exists():
            summary[run] = {"status": "pending", "file": f"results/{rel}"}
            continue
        preds = json.loads(path.read_text())["predictions"]
        c = Counter()
        types = Counter()
        self_cons = []
        for p in preds:
            parsed = p.get("parsed") or {}
            raw = p.get("raw_text", "")
            text = justification_text(parsed, raw)
            official = score_justification(text, recs[p["id"]], p.get("pred"))
            deltas = measured_deltas(recs[p["id"]])
            c["outputs"] += 1
            c["unclosed_json"] += not raw.rstrip().endswith(("}", "```"))
            c["fallback_parse"] += parsed.get("format") == "fallback"
            if official["self_consistent"] is not None:
                self_cons.append(official["self_consistent"])
            n_scored = 0
            for cl in claims_with_context(text):
                c["mentions"] += 1
                if cl["direction"] is None:
                    c["unscored_no_direction"] += 1
                    continue
                d = deltas.get(cl["indicator"])
                truth = direction_of(d, cl["indicator"])
                if truth is None:
                    c["unscored_no_measurement"] += 1
                    continue
                n_scored += 1
                ok = cl["direction"] == truth
                win = cl["window"]
                if not ok:
                    if truth == "stable":
                        same = (cl["direction"] == "increase" and d > 0) or (cl["direction"] == "decrease" and d < 0)
                        kind = "within_deadband_same_sign" if same else "within_deadband_opposite_sign"
                    elif cl["direction"] == "stable":
                        kind = "said_stable_beyond_deadband"
                    else:
                        kind = "beyond_deadband_opposite_sign"
                    types[kind] += 1
                else:
                    kind = ""
                flags = {"negation_nearby": bool(NEG.search(win)), "hedge_nearby": bool(HEDGE.search(win)),
                         "from_rule_line": bool(RULE_COPY.match(text[cl["start"]: cl["start"] + 70]))}
                for f, v in flags.items():
                    c[f] += v
                c["scored"] += 1
                c["consistent"] += ok
                table.append([run, p["id"], cl["indicator"], cl["direction"], truth, round(d, 4), int(ok), kind,
                              int(flags["negation_nearby"]), int(flags["hedge_nearby"]), int(flags["from_rule_line"]),
                              parsed.get("format", "")])
            assert n_scored == official["n_claims"], (run, p["id"], n_scored, official["n_claims"])
        rule_line = [r for r in table if r[0] == run and r[10]]
        summary[run] = {
            "file": f"results/{rel}", **dict(c),
            "inconsistent": c["scored"] - c["consistent"],
            "consistency": c["consistent"] / c["scored"] if c["scored"] else None,
            "scored_excluding_rule": sum(1 for r in table if r[0] == run and not r[10]),
            "consistent_excluding_rule": sum(r[6] for r in table if r[0] == run and not r[10]),
            "consistency_excluding_rule_lines": (
                sum(r[6] for r in table if r[0] == run and not r[10]) /
                max(1, sum(1 for r in table if r[0] == run and not r[10]))),
            "rule_line_claims": len(rule_line),
            "inconsistency_types": dict(types),
            "self_consistency": [sum(self_cons), len(self_cons)],
        }
    write_csv("claims.csv", ["run", "patch_id", "indicator", "claimed", "measured", "delta", "consistent", "error_type",
                             "negation_nearby", "hedge_nearby", "from_rule_line", "parse_format"], table)
    write_json("claim_audit.json", summary)
    for run, s in summary.items():
        print(run, {k: v for k, v in s.items() if k != "file"})


if __name__ == "__main__":
    main()
