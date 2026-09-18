"""Unified pre-flight checks before scaling EcoExplain-VLM to full benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .export_utils import inventory_patches
from .paths import (
    DATASET_JSONL,
    FINAL_LABELS_CSV,
    PARENT_SELECTED_PATCHES_CSV,
    PATCH_SCALARS_JSON,
    PROCESSED,
    PROJECT_ROOT,
    RAW_TIFFS,
    RESULTS,
    SELECTED_PATCHES_CSV,
    TRIPLETS_JSON,
)
from .validate_dataset import (
    ROADMAP_JSONL_BLOCKS,
    ROADMAP_S1_POLS,
    ROADMAP_S2_BANDS,
    _has_coordinate_leak,
    load_patch_ids,
    validate_jsonl,
    validate_raw_tiffs,
    validate_scalars,
)

Status = Literal["pass", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _load_final_labels(path: Path = FINAL_LABELS_CSV) -> "Any":
    import pandas as pd

    if not path.exists():
        return None
    return pd.read_csv(path)


def check_labels(root: Path = PROJECT_ROOT) -> list[CheckResult]:
    results: list[CheckResult] = []
    fl = _load_final_labels()
    if fl is None:
        results.append(CheckResult("final_labels.csv", "fail", f"Missing {FINAL_LABELS_CSV}"))
        return results

    n = len(fl)
    validated = int((fl["validation_status"] == "validated").sum()) if "validation_status" in fl.columns else n
    results.append(
        CheckResult(
            "human_validation",
            "pass" if validated == n else "fail",
            f"{validated}/{n} patches validated in final_labels.csv",
            {"validated": validated, "total": n},
        )
    )

    if "candidate_class" in fl.columns and "reference_class" in fl.columns:
        agree = int((fl["candidate_class"] == fl["reference_class"]).sum())
        results.append(
            CheckResult(
                "candidate_reference_agreement",
                "pass",
                f"{agree}/{n} ({round(100*agree/n,1)}%) candidate vs reference agreement",
                {"agree": agree, "total": n},
            )
        )

    child_sp = SELECTED_PATCHES_CSV
    parent_sp = PARENT_SELECTED_PATCHES_CSV
    if child_sp.exists() and parent_sp.exists():
        import pandas as pd

        c_ids = set(pd.read_csv(child_sp)["patch_id"].astype(str))
        p_ids = set(pd.read_csv(parent_sp)["patch_id"].astype(str))
        sync = c_ids == p_ids
        results.append(
            CheckResult(
                "selected_patches_sync",
                "pass" if sync else "warn",
                "Child and parent selected_patches.csv patch IDs match" if sync else "Patch ID mismatch between repos",
                {"child_only": sorted(c_ids - p_ids)[:5], "parent_only": sorted(p_ids - c_ids)[:5]},
            )
        )
    return results


def check_raw_exports(root: Path = PROJECT_ROOT, patch_ids: list[str] | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    patch_ids = patch_ids or load_patch_ids()
    rt = validate_raw_tiffs(RAW_TIFFS, patch_ids)
    complete = rt["complete_patches"] == rt["patch_count"] and rt["patch_count"] > 0
    results.append(
        CheckResult(
            "raw_geotiffs",
            "pass" if complete else "fail",
            f"{rt['complete_patches']}/{rt['patch_count']} patches have {rt['tif_per_patch_expected']} GeoTIFFs each",
            {"incomplete": rt["patch_count"] - rt["complete_patches"]},
        )
    )

    demo = list(RAW_TIFFS.glob("demo_*"))
    if demo:
        results.append(
            CheckResult(
                "demo_artifacts",
                "warn",
                f"{len(demo)} demo_kanha files in raw_tiffs/ — exclude from benchmark",
                {"count": len(demo)},
            )
        )

    if not TRIPLETS_JSON.exists():
        results.append(CheckResult("triplets.json", "fail", f"Missing {TRIPLETS_JSON}"))
    else:
        trips = json.loads(TRIPLETS_JSON.read_text())
        trip_ids = {t["scene_id"] for t in trips}
        missing = set(patch_ids) - trip_ids
        results.append(
            CheckResult(
                "triplets.json",
                "pass" if not missing else "fail",
                f"{len(trips)} DIP records; expected {len(patch_ids)}",
                {"missing_ids": sorted(missing)[:10]},
            )
        )
    return results


def check_jsonl_integrity(root: Path = PROJECT_ROOT, patch_ids: list[str] | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    patch_ids = patch_ids or load_patch_ids()
    js = validate_jsonl(DATASET_JSONL)
    if "error" in js:
        results.append(CheckResult("dataset.jsonl", "fail", js["error"]))
        return results

    schema_ok = not js.get("schema_errors")
    results.append(
        CheckResult(
            "jsonl_schema",
            "pass" if schema_ok else "fail",
            f"{js['records']} records; {len(js.get('schema_errors', []))} schema errors",
            {"errors": js.get("schema_errors", [])[:5]},
        )
    )

    ollama = js.get("ollama_vs_fallback", {})
    n_ollama = ollama.get("ollama", 0)
    n_fb = ollama.get("fallback", 0)
    results.append(
        CheckResult(
            "ollama_explanations",
            "pass" if n_fb == 0 and n_ollama == js["records"] else "warn",
            f"ollama={n_ollama}, fallback={n_fb}",
            ollama,
        )
    )

    rows = [json.loads(line) for line in DATASET_JSONL.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("id") != "demo_kanha"]

    fl = _load_final_labels()
    if fl is not None:
        ref_csv = dict(zip(fl["patch_id"].astype(str), fl["reference_class"]))
        mismatches = [
            r["id"] for r in rows if ref_csv.get(r["id"]) != r.get("reference_class")
        ]
        results.append(
            CheckResult(
                "reference_class_sync",
                "pass" if not mismatches else "fail",
                "JSONL reference_class matches final_labels.csv"
                if not mismatches
                else f"{len(mismatches)} reference_class mismatches",
                {"mismatches": mismatches[:10]},
            )
        )

    leaks = sum(1 for r in rows if _has_coordinate_leak(r))
    results.append(
        CheckResult(
            "coordinate_leak",
            "pass" if leaks == 0 else "fail",
            f"{leaks} records leak lat/lon to VLM-facing JSON",
        )
    )

    missing_paths: list[str] = []
    bad_s1 = 0
    for r in rows:
        for key in ("t0", "t1"):
            p = root / r["images"][key]
            if not p.exists():
                missing_paths.append(str(p))
        for key in ("t0", "t1"):
            p = root / r["sentinel1"][key]
            if not p.exists():
                missing_paths.append(str(p))
            else:
                arr = np.load(p)
                if arr.shape != (512, 512, 2) or not np.isfinite(arr).all():
                    bad_s1 += 1

    results.append(
        CheckResult(
            "core_asset_paths",
            "pass" if not missing_paths else "fail",
            "All T0/T1 JPGs and S1 tensors exist"
            if not missing_paths
            else f"{len(missing_paths)} missing core asset paths",
            {"sample_missing": missing_paths[:5]},
        )
    )
    results.append(
        CheckResult(
            "sar_tensor_sanity",
            "pass" if bad_s1 == 0 else "fail",
            f"S1 tensors are 512×512×2 finite arrays ({bad_s1} bad)",
        )
    )

    trip_ids = set()
    if TRIPLETS_JSON.exists():
        trip_ids = {t["scene_id"] for t in json.loads(TRIPLETS_JSON.read_text())}
    jsonl_ids = {r["id"] for r in rows}
    id_sync = jsonl_ids == trip_ids
    results.append(
        CheckResult(
            "jsonl_triplets_id_sync",
            "pass" if id_sync else "fail",
            "dataset.jsonl IDs match triplets.json"
            if id_sync
            else f"jsonl-only={sorted(jsonl_ids - trip_ids)[:3]} trips-only={sorted(trip_ids - jsonl_ids)[:3]}",
        )
    )
    return results


def check_spatial_policy(root: Path = PROJECT_ROOT) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not DATASET_JSONL.exists():
        return results
    rows = [json.loads(line) for line in DATASET_JSONL.read_text().splitlines() if line.strip()]

    stable_with_boxes = [
        r["id"] for r in rows if r.get("reference_class") == "stable" and (r.get("triplets") or [])
    ]
    decline_no_boxes = [
        r["id"] for r in rows if r.get("reference_class") == "decline" and not (r.get("triplets") or [])
    ]
    total_boxes = sum(len(r.get("triplets") or []) for r in rows)

    results.append(
        CheckResult(
            "stable_zero_box_policy",
            "pass" if not stable_with_boxes else "warn",
            "No stable patches carry Otsu boxes"
            if not stable_with_boxes
            else f"{len(stable_with_boxes)} stable patches still have boxes",
            {"ids": stable_with_boxes},
        )
    )
    results.append(
        CheckResult(
            "decline_without_boxes",
            "warn" if decline_no_boxes else "pass",
            f"{len(decline_no_boxes)} decline patches have 0 Otsu boxes (patch vs pixel mismatch)"
            if decline_no_boxes
            else "All decline patches have at least one box",
            {"ids": decline_no_boxes},
        )
    )
    results.append(
        CheckResult(
            "otsu_box_count",
            "pass",
            f"{total_boxes} total boxes; mean {round(total_boxes/len(rows),2)}/patch",
        )
    )
    return results


def check_scalars(root: Path = PROJECT_ROOT, patch_ids: list[str] | None = None) -> list[CheckResult]:
    patch_ids = patch_ids or load_patch_ids()
    ps = validate_scalars(PATCH_SCALARS_JSON, patch_ids)
    if "error" in ps:
        return [
            CheckResult(
                "patch_scalars.json",
                "warn",
                f"Optional file missing: {ps['error']} (scalars still in selected_patches.csv / triplets)",
            )
        ]
    nulls = ps.get("patches_with_null_scalars", 0)
    return [
        CheckResult(
            "patch_scalars.json",
            "pass" if nulls == 0 else "warn",
            f"{nulls} patches with null scalar keys in patch_scalars.json",
            {"issues": ps.get("issues", [])},
        )
    ]


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    root = root or PROJECT_ROOT
    patch_ids = load_patch_ids()

    checks: list[CheckResult] = []
    checks.extend(check_labels(root))
    checks.extend(check_raw_exports(root, patch_ids))
    checks.extend(check_scalars(root, patch_ids))
    checks.extend(check_jsonl_integrity(root, patch_ids))
    checks.extend(check_spatial_policy(root))

    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    ready = len(fails) == 0

    report = {
        "ready_for_scale": ready,
        "summary": {
            "pass": sum(1 for c in checks if c.status == "pass"),
            "warn": len(warns),
            "fail": len(fails),
        },
        "checks": [
            {"name": c.name, "status": c.status, "message": c.message, "details": c.details}
            for c in checks
        ],
        "blockers": [c.name for c in fails],
        "warnings": [c.name for c in warns],
    }
    return report


def print_preflight_report(report: dict[str, Any]) -> None:
    print("=" * 70)
    print("EcoExplain-VLM PRE-FLIGHT CHECK")
    print("=" * 70)
    for chk in report["checks"]:
        icon = {"pass": "OK", "warn": "!!", "fail": "XX"}[chk["status"]]
        print(f"[{icon}] {chk['name']}: {chk['message']}")
    print("-" * 70)
    s = report["summary"]
    print(f"PASS {s['pass']} | WARN {s['warn']} | FAIL {s['fail']}")
    if report["ready_for_scale"]:
        print("VERDICT: READY TO SCALE (warnings OK to review)")
    else:
        print("VERDICT: NOT READY — fix blockers:", ", ".join(report["blockers"]))
    print("=" * 70)


def write_preflight_report(report: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (RESULTS / "preflight_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out
