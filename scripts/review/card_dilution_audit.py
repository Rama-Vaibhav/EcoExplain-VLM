#!/usr/bin/env python3
"""C0.2 card-dilution audit (sir's Major 8 geometry/nodata question; plan A2).

The evidence card takes index and SAR means over the zero-padded 512x512 canvas
(scripts/03_build_dataset.py -> src/dip_utils.py::_pad_to_square). This script recomputes the
same quantities over the native patch pixels from data/raw_tiffs/ and compares them.

    .venv/bin/python scripts/review/card_dilution_audit.py
Outputs: results/review/card_dilution.csv, results/review/card_dilution_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, rows, rule, split_ids, write_csv, write_json  # noqa: E402

TIF = ROOT / "data" / "raw_tiffs"
INDICES = ("NDVI", "EVI", "NDMI", "NBR")


def band_means(path: Path) -> tuple[list[float], tuple[int, int]]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        shape = (src.width, src.height)
    return [float(np.nanmean(b)) for b in arr], shape


def main() -> None:
    recs = rows()
    pool = pd.read_csv(ROOT / "data" / "raw_metadata" / "composite_stats.csv").set_index("patch_id")
    table = []
    for pid, rec in sorted(recs.items()):
        fa = rec["file_analysis"]
        i0, shape = band_means(TIF / f"{pid}_T0_indices.tif")
        i1, _ = band_means(TIF / f"{pid}_T1_indices.tif")
        s0, _ = band_means(TIF / f"{pid}_T0_s1.tif")
        s1, _ = band_means(TIF / f"{pid}_T1_s1.tif")
        row = {"patch_id": pid, "width": shape[0], "height": shape[1],
               "clipped": shape[0] < 65 or shape[1] < 60,
               "label": rec["candidate_class"], "ee_dndvi": float(pool.ndvi_change[pid])}
        for j, name in enumerate(INDICES):
            row[f"true_{name}_t0"] = i0[j]
            row[f"card_{name}_t0"] = fa["indices_t0"][name]
            row[f"true_d{name}"] = i1[j] - i0[j]
            row[f"card_d{name}"] = fa["index_delta"][name]
        row["true_VV_t0"], row["card_VV_t0"] = s0[0], fa["sar_vv_mean_t0"]
        row["true_dVV"], row["card_dVV"] = s1[0] - s0[0], fa["sar_vv_delta"]
        row["true_dVH"], row["card_dVH"] = s1[1] - s0[1], fa["sar_vh_delta"]
        table.append(row)
    df = pd.DataFrame(table)
    write_csv("card_dilution.csv", list(df.columns), df.round(5).values.tolist())

    normal = df[~df.clipped]
    test = set(split_ids("test"))
    ratio = (df.card_NDVI_t0 / df.true_NDVI_t0)

    def cls_change(col_true: str, col_card: str, lo: float, hi: float) -> int:
        return int(sum(rule(a, lo, hi) != rule(b, lo, hi) for a, b in zip(df[col_true], df[col_card])))

    summary = {
        "n": len(df),
        "clipped": {"n": int(df.clipped.sum()), "ids": df[df.clipped].patch_id.tolist(),
                    "in_test": sorted(set(df[df.clipped].patch_id) & test)},
        "typical_shape": df[["width", "height"]].mode().iloc[0].tolist(),
        "ndvi_t0_card_over_true": {"median_all": float(ratio.median()),
                                   "median_unclipped": float((normal.card_NDVI_t0 / normal.true_NDVI_t0).median()),
                                   "min_clipped": float(ratio[df.clipped].min())},
        "dndvi_card_over_true_median_unclipped": float((normal.card_dNDVI / normal.true_dNDVI).median()),
        "true_dndvi_vs_ee_dndvi": {"corr": float(np.corrcoef(df.true_dNDVI, df.ee_dndvi)[0, 1]),
                                   "max_abs_diff": float((df.true_dNDVI - df.ee_dndvi).abs().max())},
        "vv_t0_card_vs_true_median_db": [float(df.card_VV_t0.median()), float(df.true_VV_t0.median())],
        "class_changes_pm0.02_card_vs_true": cls_change("true_dNDVI", "card_dNDVI", -0.02, 0.02),
        "examples": df[df.patch_id.isin(["p00052", "p00086", "p04399"])][
            ["patch_id", "width", "height", "card_NDVI_t0", "true_NDVI_t0", "card_dNDVI", "true_dNDVI",
             "card_VV_t0", "true_VV_t0"]].round(4).to_dict("records"),
    }
    print("wrote", write_json("card_dilution_summary.json", summary))
    print(summary)


if __name__ == "__main__":
    main()
