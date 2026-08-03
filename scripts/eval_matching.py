"""
Ungameable matching eval: legend self-recovery confusion matrix.

Per-class count-recall is gamed by select_by_count's top-N (the counts match no
matter how wrong the per-dot labels are), so it cannot tell us whether the
colour/shape rework actually improved SEPARATION. This harness measures
separation directly and without labels:

  For each legend row, take its clean marker glyph, DEGRADE it to aerial scale
  (downscale + blur + noise, on a neutral background), then run it through the
  real detection + matching path (detect_dots -> assign_classes) and check
  whether it recovers ITS OWN class. Build the confusion matrix.

This exercises exactly the code the plan changes (colour anchoring, background
removal, NCC), so re-running after each stage shows an attributable delta.
Clean background isolates the SEPARATION question; background robustness is
measured separately on real aerials by diagnose_shape_accuracy.py.

Usage:
    python scripts/eval_matching.py
    python scripts/eval_matching.py --sizes 5,7,9 --trials 3

Reports, per image and per target aerial size:
  - overall self-recovery accuracy
  - within-colour accuracy (the hard part: same-colour classes)
  - the confusion pairs (true class -> wrongly-matched class)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legend import (
    parse_screenshot,
    attach_class_names,
    _SPECIES_CODES,
    _name_hue,
)
from src.classify import detect_dots, assign_classes

ROOT = Path(__file__).parent.parent
SCREENSHOTS = ROOT / "data" / "fixtures" / "screenshots"
# D + A are the tuning bracket (cleanest truth + hardest); B + C are held-out
# generalization checks. The method is per-image, so all four exercise the same
# self-calibrating path — nothing is tuned per colony.
IMAGES = ["D_raccoon_2011", "A_felicity_2012", "B_gaillard_2011", "C_northdeer_2010"]

_RNG = np.random.default_rng(0)


def _degrade(
    marker_rgb: np.ndarray, target_px: int,
    bg_patch: np.ndarray | None = None,
) -> np.ndarray:
    """
    Render a clean legend marker as it would appear on the aerial.

    Downscale so its longest side is ~target_px, blur slightly (marker
    rendering + JPEG). Place it on the background: a neutral low-saturation
    grey (isolates SEPARATION), or a real aerial patch (`bg_patch`, stresses
    BACKGROUND REMOVAL). Returns a small RGB canvas containing one dot.
    """
    h, w = marker_rgb.shape[:2]
    if h == 0 or w == 0:
        return marker_rgb
    scale = target_px / max(h, w)
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(marker_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=0.6)

    pad = max(10, target_px * 3)
    ch, cw = sh + 2 * pad, sw + 2 * pad
    if bg_patch is not None:
        canvas = cv2.resize(bg_patch, (cw, ch), interpolation=cv2.INTER_AREA).copy()
    else:
        canvas = np.full((ch, cw, 3), (122, 128, 120), np.uint8)
        canvas = np.clip(canvas + _RNG.normal(0, 4, canvas.shape).astype(np.int16),
                         0, 255).astype(np.uint8)
    # Alpha-blend the marker's own coloured pixels over the background so only
    # the marker (not a grey square) is composited — mirrors how the tool bakes
    # a dot onto the photo.
    hsv = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    fg = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 50)
    region = canvas[pad:pad + sh, pad:pad + sw]
    region[fg] = small[fg]
    canvas[pad:pad + sh, pad:pad + sw] = region
    return canvas


def _bg_patches(rgb: np.ndarray, bbox, n: int, size: int) -> list[np.ndarray]:
    """Random aerial patches (outside the dialog) for natural backgrounds."""
    H, W = rgb.shape[:2]
    bx, by, bw, bh = bbox
    out = []
    for _ in range(n * 4):
        if len(out) >= n:
            break
        x = int(_RNG.integers(0, max(1, W - size)))
        y = int(_RNG.integers(0, max(1, H - size)))
        # reject patches overlapping the dialog
        if x < bx + bw and x + size > bx and y < by + bh and y + size > by:
            continue
        out.append(rgb[y:y + size, x:x + size])
    return out or [np.full((size, size, 3), (122, 128, 120), np.uint8)]


def _prep_legend(name: str):
    """Parse a screenshot's legend and tag each row with a unique identity."""
    path = SCREENSHOTS / f"{name}.jpg"
    if not path.exists():
        return None, None, None, None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None, None, None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    entries, bbox = parse_screenshot(rgb)
    if not entries:
        return None, None, None, None
    dialog = rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    entries = attach_class_names(dialog, entries, _SPECIES_CODES)

    # Human-readable label per row (for confusion), and a unique id we can
    # recover from the matched dot (assign_classes copies species/category).
    labels = {}
    for i, e in enumerate(entries):
        labels[i] = (e.class_name or f"{e.color}/{e.shape}")
        e.species = f"ID{i}"      # unique tag -> recoverable via dot.species
        e.category = str(i)
    return entries, labels, rgb, bbox


def _predicted_id(dot) -> int | None:
    if dot is None or dot.species is None:
        return None
    try:
        return int(dot.species[2:])  # "ID<i>"
    except ValueError:
        return None


def _eval_image(name: str, sizes: list[int], trials: int, bg: str) -> None:
    entries, labels, rgb, bbox = _prep_legend(name)
    if entries is None:
        print(f"SKIP {name} — not found or no legend")
        return

    # colour-group sizes: which rows share a colour (the hard, within-colour set)
    colour_of = {i: _name_hue(e.hue) if e.hue is not None else "grey"
                 for i, e in enumerate(entries)}
    group_size: dict[str, int] = {}
    for c in colour_of.values():
        group_size[c] = group_size.get(c, 0) + 1
    within_ids = {i for i in colour_of if group_size[colour_of[i]] > 1}

    print("\n" + "=" * 74)
    print(f"{name}  —  {len(entries)} rows, "
          f"{len(within_ids)} in same-colour groups (the hard set)  [bg={bg}]")
    print("=" * 74)
    print(f"{'aerial px':>9} {'detected':>9} {'overall':>9} {'within-colour':>14}")
    print("-" * 74)

    confusion: dict[int, dict[int, int]] = {}
    for px in sizes:
        n_total = n_detected = n_correct = 0
        n_within = n_within_correct = 0
        patches = (_bg_patches(rgb, bbox, trials * len(entries), px * 5)
                   if bg == "natural" else None)
        pk = 0
        for i, e in enumerate(entries):
            if e.marker is None or e.marker.size == 0:
                continue
            for _ in range(trials):
                n_total += 1
                is_within = i in within_ids
                n_within += int(is_within)
                bg_patch = patches[pk % len(patches)] if patches else None
                pk += 1
                canvas = _degrade(e.marker, px, bg_patch)
                dots = detect_dots(canvas)
                if not dots:
                    continue  # undetectable at this size
                # take the most central / largest-quality dot
                dot = max(dots, key=lambda d: d.quality)
                assign_classes([dot], entries)
                n_detected += 1
                pred = _predicted_id(dot)
                if pred == i:
                    n_correct += 1
                    n_within_correct += int(is_within)
                elif pred is not None:
                    confusion.setdefault(i, {}).setdefault(pred, 0)
                    confusion[i][pred] += 1
                    if is_within:
                        pass  # counted in n_within, not correct
        det = f"{100*n_detected/max(n_total,1):.0f}%"
        acc = f"{100*n_correct/max(n_detected,1):.0f}%"
        win = (f"{100*n_within_correct/max(n_within,1):.0f}%"
               if n_within else "n/a")
        print(f"{px:>7}px {det:>9} {acc:>9} {win:>14}")

    # Confusion (aggregated across sizes) — most-confused first.
    print("\n  Confusion (true -> wrongly matched, aggregated over sizes):")
    flat = []
    for true_i, preds in confusion.items():
        for pred_i, cnt in preds.items():
            flat.append((cnt, true_i, pred_i))
    flat.sort(reverse=True)
    if not flat:
        print("    (none)")
    for cnt, true_i, pred_i in flat[:12]:
        tag = "  [same-colour]" if (true_i in within_ids and pred_i in within_ids
                                    and colour_of[true_i] == colour_of[pred_i]) else ""
        print(f"    {labels[true_i][:24]:24} -> {labels[pred_i][:24]:24} "
              f"x{cnt}{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="5,7,9,11",
                    help="comma-separated target aerial marker sizes (px)")
    ap.add_argument("--trials", type=int, default=3,
                    help="degradation trials per row per size")
    ap.add_argument("--bg", choices=["neutral", "natural"], default="neutral",
                    help="neutral grey (isolates separation) or real aerial "
                         "patches (stresses background removal)")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    for name in IMAGES:
        _eval_image(name, sizes, args.trials, args.bg)

    print("\n" + "=" * 74)
    print("Baseline before the rework. Re-run after each stage; 'within-colour' "
          "is the\nnumber the colour/shape work must move. Overall includes the "
          "easy\nsingle-colour classes and will look high regardless.")


if __name__ == "__main__":
    main()
