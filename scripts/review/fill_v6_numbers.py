#!/usr/bin/env python3
"""Fill the numeric «PLACEHOLDERS» in report v6 from results/review/stats_report.json and claim_audit.json.

Prints every substitution so it can be checked. Text placeholders (sentences that interpret a
result) are left for the author and listed at the end.

    .venv/bin/python scripts/review/fill_v6_numbers.py [--dry-run]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, REVIEW, ROOT  # noqa: E402

TEX = ROOT.parent / "report" / "paper_v3" / "EcoExplain_VLM_paper_v6.tex"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def p_fmt(p: float) -> str:
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.2 else f"{p:.2f}")


def main() -> None:
    s = json.loads((REVIEW / "stats_report.json").read_text())
    c = json.loads((REVIEW / "claim_audit.json").read_text())
    runs = s["runs"]
    mc = {m["test"]: m for m in s["mcnemar"] if "p_holm" in m}
    vals: dict[str, str] = {}

    def run_vals(prefix: str, key: str) -> None:
        r = runs[key]
        vals[f"{prefix}_ACC"] = pct(r["accuracy"])
        vals[f"{prefix}_K"] = str(r["k"])
        vals[f"{prefix}_RECALL"] = (f"{r['confusion']['decline']['decline']}/18 decline, "
                                    f"{r['confusion']['stable']['stable']}/18 stable and "
                                    f"{r['confusion']['increase']['increase']}/19 increase correct")

    for prefix, key in (("X1", "qwen_evid_336"), ("X2", "qwen_evid_384"), ("X3", "qwen_expl_zs_384"), ("X7", "qwen_full_rerun")):
        if runs.get(key, {}).get("status") == "ok":
            run_vals(prefix, key)
    for prefix, test in (("LEAK", "leak_primary"), ("CL", "classlora_vs_matched"), ("EL", "expllora_vs_matched")):
        if test in mc:
            m = mc[test]
            vals[f"{prefix}_A"] = str(m["a_right_b_wrong"])
            vals[f"{prefix}_B"] = str(m["b_right_a_wrong"])
            vals[f"{prefix}_P"] = p_fmt(m["p_exact_two_sided"])
            vals[f"{prefix}_PH"] = p_fmt(m["p_holm"])
    if "leak_secondary" in mc:
        vals["LEAK2_PH"] = p_fmt(mc["leak_secondary"]["p_holm"])
    if c.get("qwen_expl_zs_384", {}).get("scored"):
        z = c["qwen_expl_zs_384"]
        vals["X3_CLAIMS"] = f"{pct(z['consistent'] / z['scored'])}\\% ({z['consistent']}/{z['scored']})"

    text = TEX.read_text()
    for k, v in vals.items():
        n = text.count(f"«{k}»")
        if n:
            print(f"{k:10s} -> {v}   ({n}x)")
            if "--dry-run" not in sys.argv:
                text = text.replace(f"«{k}»", v)
    if "--dry-run" not in sys.argv:
        TEX.write_text(text)
    left = sorted(set(re.findall(r"«([A-Z0-9_]+)»", text)))
    print("left for the author:", left)


if __name__ == "__main__":
    main()
