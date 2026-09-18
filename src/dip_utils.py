"""Digital image processing for EcoExplain-VLM ground-truth extraction.

Pipeline:
  1. Otsu thresholding on ΔNDVI / ΔNBR GeoTIFFs to isolate significant drops.
  2. Contour-based bounding boxes in native chip coordinates.
  3. Zero-pad + resize RGB chips to a square VLM canvas, scaling boxes with them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import cv2
import numpy as np
import rasterio
from skimage.filters import threshold_otsu

PathLike = Union[str, Path]
BBox = list  # [x_min, y_min, x_max, y_max]


def read_single_band(tensor_path: PathLike) -> np.ndarray:
    """Read the first band of a GeoTIFF as float32, preserving nodata as NaN."""
    path = Path(tensor_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
    return arr


def process_tensor_otsu(
    tensor_path: PathLike,
    *,
    isolate_drops: bool = True,
    min_drop: float = 0.0,
) -> np.ndarray:
    """Read a single-band Δ-index GeoTIFF and return a uint8 binary mask.

    Significant *negative* change (deforestation / burn) is isolated by:
      1. Taking the magnitude of negative pixels (drops).
      2. Running Otsu's method on those drop magnitudes (skimage).
      3. Thresholding so only drops at or above the Otsu cut survive.

    If the raster has no negative values, an all-zero mask is returned.
    Positive greening is ignored when ``isolate_drops=True``.
    """
    arr = read_single_band(tensor_path)
    finite = np.isfinite(arr)

    if isolate_drops:
        drops = np.where(finite & (arr < -abs(min_drop)), -arr, 0.0).astype(np.float32)
    else:
        drops = np.where(finite, np.abs(arr), 0.0).astype(np.float32)

    positive_drops = drops[drops > 0]
    if positive_drops.size < 16:
        return np.zeros(arr.shape, dtype=np.uint8)

    # Otsu needs more than one unique value.
    if np.unique(positive_drops).size < 2:
        thresh = float(positive_drops.max())
    else:
        thresh = float(threshold_otsu(positive_drops))

    mask = (drops >= thresh).astype(np.uint8)
    return mask


def extract_bounding_boxes(
    binary_mask: np.ndarray,
    *,
    min_area: int = 64,
    morph_kernel: int = 3,
) -> list[BBox]:
    """Find connected components and return boxes ``[x_min, y_min, x_max, y_max]``.

    Coordinates are in the *native* mask / chip pixel space (origin top-left).
    A small morphological close fills 1-pixel holes before contour detection.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2-D, got shape {binary_mask.shape}")

    mask = (binary_mask > 0).astype(np.uint8) * 255
    if morph_kernel and morph_kernel >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def _pad_to_square(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Zero-pad a H×W (×C) array to max(H, W) square. Returns image, pad_x, pad_y.

    Padding is split as evenly as possible on both sides (black borders).
    """
    h, w = image.shape[:2]
    side = max(h, w)
    pad_y = side - h
    pad_x = side - w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    if image.ndim == 2:
        padded = np.pad(image, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    else:
        padded = np.pad(
            image,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return padded, left, top


def pad_and_resize(
    image_array: np.ndarray,
    target_size: tuple[int, int] = (512, 512),
    bboxes: Optional[Sequence[Sequence[int]]] = None,
    keep_dtype: bool = False,
) -> tuple[np.ndarray, list[BBox], dict]:
    """Pad an RGB (or single-band) chip to square, then resize to ``target_size``.

    Aspect ratio is preserved with zero-padding (black borders). Bounding boxes
    given in native chip coordinates are mapped onto the resized canvas.

    Returns
    -------
    resized : np.ndarray
        uint8 image of shape (target_h, target_w, C) or (target_h, target_w).
    scaled_bboxes : list of [x_min, y_min, x_max, y_max]
        Boxes in the *final* VLM image coordinate system.
    meta : dict
        Transform parameters (pads, scale, original shape) for debugging.
    """
    if image_array.ndim not in (2, 3):
        raise ValueError(f"image_array must be 2-D or 3-D, got {image_array.ndim}-D")

    orig_h, orig_w = image_array.shape[:2]
    target_w, target_h = int(target_size[0]), int(target_size[1])

    padded, pad_x, pad_y = _pad_to_square(image_array)
    square_side = padded.shape[0]

    interp = cv2.INTER_LINEAR if image_array.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(padded, (target_w, target_h), interpolation=interp)

    # Display-ready uint8 unless the caller needs raw physical units (S1, DEM, indices).
    if not keep_dtype and resized.dtype != np.uint8:
        finite = np.isfinite(resized)
        work = np.nan_to_num(resized, nan=0.0)
        if work.max() <= 1.5:
            work = np.clip(work, 0, 1) * 255.0
        else:
            lo, hi = np.percentile(work[finite] if finite.any() else work, (2, 98))
            if hi <= lo:
                hi = lo + 1.0
            work = np.clip((work - lo) / (hi - lo), 0, 1) * 255.0
        resized = work.astype(np.uint8)

    scale_x = target_w / float(square_side)
    scale_y = target_h / float(square_side)

    scaled: list[BBox] = []
    if bboxes:
        for box in bboxes:
            x0, y0, x1, y1 = [float(v) for v in box]
            nx0 = (x0 + pad_x) * scale_x
            ny0 = (y0 + pad_y) * scale_y
            nx1 = (x1 + pad_x) * scale_x
            ny1 = (y1 + pad_y) * scale_y
            scaled.append(
                [
                    int(np.clip(round(nx0), 0, target_w - 1)),
                    int(np.clip(round(ny0), 0, target_h - 1)),
                    int(np.clip(round(nx1), 1, target_w)),
                    int(np.clip(round(ny1), 1, target_h)),
                ]
            )

    meta = {
        "orig_hw": (orig_h, orig_w),
        "pad_x": pad_x,
        "pad_y": pad_y,
        "square_side": square_side,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "target_size": (target_w, target_h),
    }
    return resized, scaled, meta


def percentile_stretch_rgb(
    rgb: np.ndarray,
    vmax: float = 0.22,
    gamma: float = 1.0,
    **_ignored: float,
) -> np.ndarray:
    """Natural-color uint8 RGB for Sentinel-2 true-color chips.

    Uses the standard fixed reflectance window (B4/B3/B2 ÷ vmax), matching
    GEE's typical ``min: 0, max: 2200`` on a 0–10000 scale. This keeps
    forest green/brown without per-band purple stretch or neon yellow-green
    from extra gain on tiny patches.
    """
    bands = np.asarray(rgb[..., :3], dtype=np.float32)
    if bands.size == 0:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)

    if np.nanmax(bands) > 1.5:
        bands = bands / 10000.0
    bands = np.where(np.isfinite(bands), bands, 0.0)
    bands = np.clip(bands, 0.0, None)

    if vmax <= 0:
        vmax = 0.22

    scaled = np.clip(bands / vmax, 0.0, 1.0)
    if gamma != 1.0:
        scaled = scaled ** (1.0 / gamma)

    return (scaled * 255.0).astype(np.uint8)


def load_rgb_geotiff(path: PathLike) -> np.ndarray:
    """Load a 3-band (or more) GeoTIFF as H×W×3 float32 RGB (first three bands)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RGB GeoTIFF not found: {path}")
    with rasterio.open(path) as src:
        count = src.count
        if count >= 3:
            data = src.read([1, 2, 3]).astype(np.float32)
            rgb = np.transpose(data, (1, 2, 0))
        else:
            band = src.read(1).astype(np.float32)
            rgb = np.stack([band, band, band], axis=-1)
        nodata = src.nodata
        if nodata is not None:
            rgb = np.where(rgb == nodata, np.nan, rgb)
    return rgb


def _mean_drop_in_box(tensor: np.ndarray, box: Sequence[int]) -> float:
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = tensor.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = tensor[y0:y1, x0:x1]
    crop = crop[np.isfinite(crop)]
    if crop.size == 0:
        return 0.0
    return float(-np.nanmean(np.minimum(crop, 0.0)))


def _severity_from_drop(mean_drop: float, high: float, moderate: float) -> str:
    if mean_drop >= high:
        return "High"
    if mean_drop >= moderate:
        return "Moderate"
    return "Low"


def build_disturbance_triplets(
    ndvi_mask: np.ndarray,
    nbr_mask: np.ndarray,
    delta_ndvi: np.ndarray,
    delta_nbr: np.ndarray,
    *,
    ndvi_boxes: Optional[Iterable[Sequence[int]]] = None,
    nbr_boxes: Optional[Iterable[Sequence[int]]] = None,
    iou_merge: float = 0.4,
) -> list[list]:
    """Build ``[[x_min, y_min, x_max, y_max], type, severity]`` triplets.

    Overlapping fire (ΔNBR) and deforestation (ΔNDVI) boxes are merged: if IoU
    exceeds ``iou_merge``, the disturbance is labelled Fire (burn typically
    also drops NDVI). Otherwise each box keeps its source type.
    """
    if ndvi_boxes is None:
        ndvi_boxes = extract_bounding_boxes(ndvi_mask)
    if nbr_boxes is None:
        nbr_boxes = extract_bounding_boxes(nbr_mask)

    def iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter == 0:
            return 0.0
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    used_ndvi = set()
    triplets: list[list] = []

    for fire_box in nbr_boxes:
        matched = None
        for i, defo_box in enumerate(ndvi_boxes):
            if i in used_ndvi:
                continue
            if iou(fire_box, defo_box) >= iou_merge:
                matched = i
                break
        if matched is not None:
            used_ndvi.add(matched)
            x0 = min(fire_box[0], ndvi_boxes[matched][0])
            y0 = min(fire_box[1], ndvi_boxes[matched][1])
            x1 = max(fire_box[2], ndvi_boxes[matched][2])
            y1 = max(fire_box[3], ndvi_boxes[matched][3])
            box = [x0, y0, x1, y1]
        else:
            box = list(fire_box)
        drop = _mean_drop_in_box(delta_nbr, box)
        triplets.append([box, "Fire", _severity_from_drop(drop, high=0.27, moderate=0.10)])

    for i, defo_box in enumerate(ndvi_boxes):
        if i in used_ndvi:
            continue
        box = list(defo_box)
        drop = _mean_drop_in_box(delta_ndvi, box)
        triplets.append([box, "Deforestation", _severity_from_drop(drop, high=0.25, moderate=0.08)])

    return triplets


def apply_reference_class_policy(
    triplets: list[list],
    reference_class: str | None,
    *,
    max_boxes: int = 5,
) -> list[list]:
    """Align Otsu spatial triplets with human patch-level reference labels.

    Otsu detects spectral *drops* only, so stable/increase patches should not
    carry many (or any) disturbance boxes. Decline patches keep the largest boxes.
    """
    if not triplets:
        return []
    ref = (reference_class or "").strip().lower()
    if ref in {"stable", "increase", "uncertain", ""}:
        return []
    if ref == "decline":
        kept = [t for t in triplets if t[1] == "Deforestation"] or list(triplets)
        kept.sort(key=lambda t: (t[2] != "High", t[2] != "Moderate", -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1])))
        return kept[:max_boxes]
    if ref == "fire":
        kept = [t for t in triplets if t[1] == "Fire"] or list(triplets)
        return kept[:max_boxes]
    return triplets[:max_boxes]


# File stems produced by notebooks/01_gee_extraction.ipynb
SCENE_SUFFIXES: dict[str, str] = {
    "t0_rgb": "_T0_rgb",
    "t1_rgb": "_T1_rgb",
    "t0_optical": "_T0_s2_optical",
    "t1_optical": "_T1_s2_optical",
    "t0_indices": "_T0_indices",
    "t1_indices": "_T1_indices",
    "t0_s1": "_T0_s1",
    "t1_s1": "_T1_s1",
    "t0_rainfall": "_T0_rainfall",
    "t1_rainfall": "_T1_rainfall",
    "t0_era5": "_T0_era5",
    "t1_era5": "_T1_era5",
    "dem": "_dem",
    "delta_ndvi": "_delta_ndvi",
    "delta_nbr": "_delta_nbr",
}

REQUIRED_DIP_KEYS = ("t0_rgb", "t1_rgb", "delta_ndvi", "delta_nbr")


def discover_scenes(raw_dir: PathLike) -> list[dict]:
    """Group GeoTIFFs in ``raw_tiffs`` by scene / patch id."""
    raw_dir = Path(raw_dir)
    scenes: dict[str, dict] = {}
    items = sorted(SCENE_SUFFIXES.items(), key=lambda kv: len(kv[1]), reverse=True)
    for tif in sorted(raw_dir.glob("*.tif")):
        name = tif.stem
        matched_key = None
        matched_suffix = None
        for key, suffix in items:
            if name.endswith(suffix):
                matched_key, matched_suffix = key, suffix
                break
        if matched_key is None:
            continue
        scene_id = name[: -len(matched_suffix)]
        rec = scenes.setdefault(scene_id, {"scene_id": scene_id})
        rec[matched_key] = tif

    complete = [
        rec for rec in scenes.values() if all(k in rec for k in REQUIRED_DIP_KEYS)
    ]
    complete.sort(key=lambda r: r["scene_id"])
    return complete


def load_geotiff_hwc(path: PathLike) -> np.ndarray:
    """Load any GeoTIFF as float32 H×W×C."""
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
    return np.transpose(data, (1, 2, 0))


def array_nanmean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))
