#!/usr/bin/env python3
"""C0.8: rebuild the candidate grid and the WorldCover forest/water filter (flow table; empty strip).

Re-implements build_candidate_grid() and flag_patch() from the pilot notebook
(.ipynb_checkpoints/Dataset_Extraction-checkpoint.ipynb, cells 12-13) and checks
(i) the grid size, (ii) that grid IDs/coordinates match the 1,200 scored candidates,
(iii) how many cells pass tree >= 0.6 and water <= 0.05, and (iv) the pass rate by longitude,
which explains the sparse strip near 80.75-80.85 E. Needs Earth Engine access.

    .venv/bin/python scripts/review/worldcover_grid_check.py
Output: results/review/worldcover_grid_check.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, write_json  # noqa: E402

import ee  # noqa: E402

PATCH_M = 640
MIN_FOREST, MAX_WATER = 0.6, 0.05


def grid(bbox: list[float]) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    m_lat = 111_320
    m_lon = 111_320 * np.cos(np.radians((min_lat + max_lat) / 2))
    step_lat, step_lon = PATCH_M / m_lat, PATCH_M / m_lon
    centers = []
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            centers.append((lon, lat))
            lon += step_lon
        lat += step_lat
    return pd.DataFrame({"patch_id": [f"p{i:05d}" for i in range(len(centers))],
                         "lon": [c[0] for c in centers], "lat": [c[1] for c in centers]})


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "pilot_kanha.yaml").read_text())
    ee.Initialize(project=cfg["project"]["gcp_project_id"])
    # The notebook took the bounds of ee.Geometry.Rectangle(roi_bbox); same numbers.
    g = grid(cfg["study_area"]["roi_bbox"])
    pool = pd.read_csv(ROOT / "data" / "raw_metadata" / "composite_stats.csv").set_index("patch_id")
    j = g.set_index("patch_id").loc[pool.index]
    coord_err = float(np.hypot(j.lon - pool.lon, j.lat - pool.lat).max())

    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    tree, water = wc.eq(10).rename("tree"), wc.eq(80).rename("water")
    img = tree.addBands(water)
    fracs = []
    for start in range(0, len(g), 400):
        chunk = g.iloc[start:start + 400]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([r.lon, r.lat]).buffer(PATCH_M / 2).bounds(), {"pid": r.patch_id})
            for r in chunk.itertuples()
        ])
        res = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4).getInfo()
        fracs += [(f["properties"]["pid"], f["properties"].get("tree"), f["properties"].get("water"))
                  for f in res["features"]]
        print("done", start + len(chunk), "/", len(g), flush=True)
    f = pd.DataFrame(fracs, columns=["patch_id", "tree", "water"]).set_index("patch_id")
    g = g.set_index("patch_id").join(f)
    g["passes"] = (g.tree >= MIN_FOREST) & (g.water <= MAX_WATER)
    g.round(4).to_csv(ROOT / "results" / "review" / "worldcover_grid.csv")

    bands = pd.cut(g.lon, bins=np.arange(80.40, 81.10, 0.05))
    by_lon = g.groupby(bands, observed=True).agg(cells=("passes", "size"), pass_rate=("passes", "mean"),
                                                 tree_mean=("tree", "mean"))
    out = {
        "grid_cells": len(g),
        "pool_ids_on_grid": int(pool.index.isin(g.index).sum()),
        "pool_coordinate_max_error_deg": coord_err,
        "passing_filter": int(g.passes.sum()),
        "pilot_notebook_reported": 3668,
        "pool_all_pass": bool(g.loc[pool.index, "passes"].all()),
        "pool_failing_ids": g.loc[pool.index][~g.loc[pool.index, "passes"]].index.tolist()[:20],
        "strip_80.75_80.85": {
            "cells": int(((g.lon >= 80.75) & (g.lon < 80.85)).sum()),
            "pass_rate": float(g[(g.lon >= 80.75) & (g.lon < 80.85)].passes.mean()),
            "tree_mean": float(g[(g.lon >= 80.75) & (g.lon < 80.85)].tree.mean()),
        },
        "rest_pass_rate": float(g[(g.lon < 80.75) | (g.lon >= 80.85)].passes.mean()),
        "by_longitude": {str(k): {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in by_lon.iterrows()},
    }
    print("wrote", write_json("worldcover_grid_check.json", out))
    print({k: v for k, v in out.items() if k != "by_longitude"})


if __name__ == "__main__":
    main()
