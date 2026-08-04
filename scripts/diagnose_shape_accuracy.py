"""
Diagnostic: aerial shape accuracy for dense colony A.

Checks two things the advisor flagged:
1. NEIGHBOR CONTAMINATION — after cluster splitting, what fraction of dot crops
   contain more than one colored blob? If high, shape errors come from
   contamination, not misclassification.
2. SHAPE DISTRIBUTION — what mix of shapes does detect_dots report, vs. what
   the legend expects?

Usage:
    python scripts/diagnose_shape_accuracy.py

Reads:  data/fixtures/screenshots/A_felicity_2012.jpg
Output: printed summary + scripts/shape_debug/sample_crops_*.png (BGR)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legend import parse_screenshot, attach_class_names, _SPECIES_CODES
from src.classify import detect_dots, assign_classes, _dot_centers

IMG_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "screenshots" / "A_felicity_2012.jpg"
OUT_DIR = Path(__file__).parent / "shape_debug"


def _count_colored_blobs_in_crop(crop_rgb: np.ndarray, sat_min: int = 60) -> int:
    """
    How many distinct colored blobs are in a small crop?

    Used to detect neighbor contamination: if a crop contains 2+ blobs, the
    shape classifier is reading a composite glyph, not a single dot.
    """
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > sat_min) & (v > 55)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    # Count components with area > 3px (noise filter)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3)


def _crop_at(rgb: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    H, W = rgb.shape[:2]
    y0 = max(0, int(cy) - half)
    y1 = min(H, int(cy) + half + 1)
    x0 = max(0, int(cx) - half)
    x1 = min(W, int(cx) + half + 1)
    return rgb[y0:y1, x0:x1]


def main():
    if not IMG_PATH.exists():
        print(f"SKIP — fixture not found: {IMG_PATH}")
        return

    img_bgr = cv2.imread(str(IMG_PATH))
    if img_bgr is None:
        print(f"SKIP — could not read {IMG_PATH}")
        return
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    print(f"Image: {IMG_PATH.name}  ({W}x{H})")

    # Step 1: parse legend.
    entries, bbox = parse_screenshot(rgb)
    if not entries:
        print("No legend entries found — cannot continue.")
        return
    entries = attach_class_names(rgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]],
                                 entries, _SPECIES_CODES)
    print(f"Legend: {len(entries)} rows, bbox={bbox}")

    # Legend shape distribution (what we expect to see in the aerial).
    legend_shapes: dict[str, int] = {}
    for e in entries:
        legend_shapes[e.shape] = legend_shapes.get(e.shape, 0) + 1
    print("\nLegend shape distribution:")
    for shape, n in sorted(legend_shapes.items(), key=lambda x: -x[1]):
        print(f"  {shape:12s}: {n}")

    # Step 2: detect aerial dots.
    print("\nDetecting aerial dots...")
    aerial_rgb = rgb  # full screenshot; dialog excluded by bbox
    dots = detect_dots(aerial_rgb, exclude=bbox)
    print(f"Detected {len(dots)} dots")

    # Step 3: shape distribution of detected dots.
    dot_shapes: dict[str, int] = {}
    for d in dots:
        dot_shapes[d.shape] = dot_shapes.get(d.shape, 0) + 1
    print("\nDetected dot shape distribution:")
    for shape, n in sorted(dot_shapes.items(), key=lambda x: -x[1]):
        pct = 100 * n / max(len(dots), 1)
        print(f"  {shape:12s}: {n:4d}  ({pct:.0f}%)")

    # Step 4: neighbor contamination check.
    # Re-compute single_area and half from _dot_centers.
    _, single_area = _dot_centers(aerial_rgb, exclude=bbox)
    half = max(4, int(round(np.sqrt(single_area) * 0.9)))
    print(f"\nCrop half-size: {half}px (single_area={single_area:.0f})")

    n_contaminated = 0
    n_clean = 0
    blob_counts: dict[int, int] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, dot in enumerate(dots):
        crop = _crop_at(aerial_rgb, dot.cx, dot.cy, half)
        if crop.size == 0:
            continue
        n_blobs = _count_colored_blobs_in_crop(crop)
        blob_counts[n_blobs] = blob_counts.get(n_blobs, 0) + 1
        if n_blobs > 1:
            n_contaminated += 1
        else:
            n_clean += 1

        # Save a sample of each contamination level (first 6 per category).
        if n_blobs >= 1 and blob_counts.get(n_blobs, 0) <= 6:
            big = cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                (crop.shape[1] * 6, crop.shape[0] * 6),
                interpolation=cv2.INTER_NEAREST,
            )
            fname = OUT_DIR / f"blobs{n_blobs}_shape{dot.shape}_dot{i:04d}.png"
            cv2.imwrite(str(fname), big)

    total = n_contaminated + n_clean
    print(f"\nCrop contamination (blobs > 1 = neighbor present):")
    for n, cnt in sorted(blob_counts.items()):
        pct = 100 * cnt / max(total, 1)
        tag = " <-- neighbor contamination" if n > 1 else ""
        print(f"  {n} blob(s): {cnt:4d} crops ({pct:.0f}%){tag}")
    print(f"  Contaminated (>1 blob): {n_contaminated}/{total} = "
          f"{100*n_contaminated//max(total,1)}%")

    # Step 5: assign classes and compare shape assignment.
    dots = assign_classes(dots, entries)
    assigned = [d for d in dots if d.class_name]
    unassigned = [d for d in dots if not d.class_name]
    print(f"\nClass assignment: {len(assigned)} assigned, {len(unassigned)} unassigned")

    # Show per-class count vs. legend count.
    class_counts: dict[str, int] = {}
    for d in assigned:
        key = d.class_name or "?"
        class_counts[key] = class_counts.get(key, 0) + 1
    print("\nDetected vs. legend counts (assigned dots, no top-N selection):")
    print(f"  {'class':30s} {'detected':>8} {'legend':>8}")
    for e in entries:
        key = e.class_name or f"{e.color}/{e.shape}"
        detected = class_counts.get(key, 0)
        legend_cnt = e.count if e.count is not None else "?"
        diff = ""
        if isinstance(legend_cnt, int) and legend_cnt > 0:
            recall = 100 * detected / legend_cnt
            diff = f"  ({recall:.0f}% recall)"
        print(f"  {key:30s} {detected:>8} {str(legend_cnt):>8}{diff}")

    print(f"\nSample crops saved to: {OUT_DIR}")
    print("  blobs1_* = clean single dot  |  blobs2+_* = contaminated crop")


if __name__ == "__main__":
    main()
