#!/usr/bin/env python3
"""C0.7 geometry and spatial-dependence audit (sir's Major 3.4; statistical check on spatial dependence).

Footprint sizes, neighbouring footprints and whether they cross split boundaries, the closest
test-train distance under each split, and how strongly the classes cluster in space
(nearest-neighbour same-class share against a label-permutation null).

    .venv/bin/python scripts/review/geometry_audit.py
Output: results/review/geometry_audit.json
"""

from __future__ import annotations

import math
import random
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, split_ids, write_json  # noqa: E402

FOOT = 640.0  # side of the square footprint in metres (buffer(320 m).bounds())


def xy(lon: float, lat: float) -> tuple[float, float]:
    return lon * 111_320 * math.cos(math.radians(22.28)), lat * 111_320


def main() -> None:
    sel = pd.read_csv(ROOT / "data" / "raw_metadata" / "selected_patches.csv").set_index("patch_id")
    pts = {p: xy(r.lon, r.lat) for p, r in sel.iterrows()}
    cls = sel.candidate_class.to_dict()
    pairs = []
    for a, b in combinations(sel.index, 2):
        dx, dy = abs(pts[a][0] - pts[b][0]), abs(pts[a][1] - pts[b][1])
        if dx < FOOT + 5 and dy < FOOT + 5:
            pairs.append((a, b, math.hypot(dx, dy), "edge" if min(dx, dy) < 50 else "corner"))
    out = {"footprint_m": FOOT, "touching_pairs": len(pairs),
           "edge": sum(p[3] == "edge" for p in pairs), "corner": sum(p[3] == "corner" for p in pairs),
           "same_class_touching": sum(cls[a] == cls[b] for a, b, *_ in pairs)}
    for scheme in ("random", "spatial"):
        role = {p: s for s in ("train", "dev", "test") for p in split_ids(s, scheme)}
        cross = [(a, b) for a, b, *_ in pairs if role[a] != role[b]]
        test = [p for p in role if role[p] == "test"]
        train = [p for p in role if role[p] == "train"]
        nearest = min(math.dist(pts[a], pts[b]) for a in test for b in train)
        out[scheme] = {"touching_pairs_crossing_splits": len(cross),
                       "crossing_pairs_with_test": sum("test" in (role[a], role[b]) for a, b in cross),
                       "test_patches_touching_train_or_dev": len({x for a, b in cross for x in (a, b)
                                                                   if role[x] == "test"}),
                       "min_test_train_distance_m": nearest}

    # Spatial clustering of labels: share of patches whose nearest neighbour has the same class.
    ids = list(sel.index)
    nn = {a: min((b for b in ids if b != a), key=lambda b: math.dist(pts[a], pts[b])) for a in ids}
    observed = sum(cls[a] == cls[nn[a]] for a in ids) / len(ids)
    rng = random.Random(42)
    labels = [cls[a] for a in ids]
    null = []
    for _ in range(2000):
        rng.shuffle(labels)
        perm = dict(zip(ids, labels))
        null.append(sum(perm[a] == perm[nn[a]] for a in ids) / len(ids))
    out["nn_same_class"] = {"observed": observed, "null_mean": sum(null) / len(null),
                            "null_p95": sorted(null)[int(0.95 * len(null))],
                            "perm_p_one_sided": (1 + sum(v >= observed for v in null)) / (1 + len(null))}
    print("wrote", write_json("geometry_audit.json", out))
    print(out)


if __name__ == "__main__":
    main()
