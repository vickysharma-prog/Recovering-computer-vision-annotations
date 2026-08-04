"""
Step 0 diagnostic (read-only): measure the real legend marker colours.

Josh's ask is to anchor aerial-dot colour to the dialog's own palette, and to
separate same-colour/different-class pairs (light red vs dark red). Before
building any colour-space machinery we measure, from the data, WHICH colour
space and dimension actually separates those pairs:

  - Parse each study screenshot's legend (parse_screenshot + attach_class_names).
  - For every legend row, read the marker glyph's mean HSV and Lab from its own
    coloured pixels.
  - Within each coarse colour group (>1 row), print the pairwise distance in
    HSV vs Lab and the per-dimension deltas, so we can see whether hue, value,
    or a*/b* is the separating axis.

Decision this unblocks: HSV-with-value vs Lab for the anchoring space.

Usage:
    python scripts/diagnose_marker_colors.py

Reads:  data/fixtures/screenshots/{A_felicity_2012,D_raccoon_2011}.jpg
Prints: per-row colour table + within-colour pairwise separation.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legend import (
    parse_screenshot,
    attach_class_names,
    _SPECIES_CODES,
    _SAT_MIN,
    _VAL_MIN,
    _name_hue,
    _circular_mean_hue,
)

ROOT = Path(__file__).parent.parent
SCREENSHOTS = ROOT / "data" / "fixtures" / "screenshots"
IMAGES = ["D_raccoon_2011", "A_felicity_2012"]


def _marker_color(marker_rgb: np.ndarray) -> dict | None:
    """
    Mean HSV + Lab of a marker glyph, from its own coloured pixels.

    Returns None when the marker has no saturated pixels (a grey marker).
    """
    if marker_rgb.size == 0:
        return None
    bgr = cv2.cvtColor(marker_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    fg = (s > _SAT_MIN) & (v > _VAL_MIN)
    if int(fg.sum()) < 3:
        return None
    hue = _circular_mean_hue(hsv[:, :, 0][fg])
    return {
        "n": int(fg.sum()),
        "h": hue,
        "s": float(s[fg].mean()),
        "v": float(v[fg].mean()),
        "L": float(lab[:, :, 0][fg].mean()),
        "a": float(lab[:, :, 1][fg].mean()),
        "b": float(lab[:, :, 2][fg].mean()),
    }


def _hue_delta(h1: float, h2: float) -> float:
    """Circular hue distance on OpenCV's 0-180 scale."""
    d = abs(h1 - h2) % 180.0
    return min(d, 180.0 - d)


def _hsv_dist(c1: dict, c2: dict) -> float:
    """Euclidean distance in (hue*2 circular, sat, val) — hue scaled to ~degrees."""
    dh = _hue_delta(c1["h"], c2["h"]) * 2.0  # 0-180 -> 0-360 like sat/val ranges
    return float(np.sqrt(dh ** 2 + (c1["s"] - c2["s"]) ** 2 + (c1["v"] - c2["v"]) ** 2))


def _lab_dist(c1: dict, c2: dict) -> float:
    """CIE76 Lab distance (OpenCV 8-bit Lab)."""
    return float(np.sqrt(
        (c1["L"] - c2["L"]) ** 2 + (c1["a"] - c2["a"]) ** 2 + (c1["b"] - c2["b"]) ** 2
    ))


def _label(e) -> str:
    if e.class_name:
        return e.class_name
    parts = [p for p in (e.species, e.category) if p]
    return " ".join(parts) if parts else f"row{e.row}"


def run(name: str) -> None:
    path = SCREENSHOTS / f"{name}.jpg"
    if not path.exists():
        print(f"SKIP {name} — not found: {path}")
        return
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"SKIP {name} — cv2 could not read it")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    entries, bbox = parse_screenshot(rgb)
    if not entries:
        print(f"{name}: no legend entries")
        return
    dialog = rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    entries = attach_class_names(dialog, entries, _SPECIES_CODES)

    print("\n" + "=" * 78)
    print(f"{name}  ({rgb.shape[1]}x{rgb.shape[0]})  —  {len(entries)} legend rows")
    print("=" * 78)

    rows = []
    for e in entries:
        c = _marker_color(e.marker)
        rows.append((e, c))

    # Per-row colour table.
    print(f"\n{'row':>3} {'colour':>8} {'label':22} {'n':>4} "
          f"{'H':>5} {'S':>4} {'V':>4} | {'L':>4} {'a':>4} {'b':>4}")
    print("-" * 78)
    for e, c in rows:
        if c is None:
            print(f"{e.row:>3} {'grey':>8} {_label(e)[:22]:22}  (no saturated pixels)")
            continue
        print(f"{e.row:>3} {_name_hue(c['h']):>8} {_label(e)[:22]:22} {c['n']:>4} "
              f"{c['h']:>5.0f} {c['s']:>4.0f} {c['v']:>4.0f} | "
              f"{c['L']:>4.0f} {c['a']:>4.0f} {c['b']:>4.0f}")

    # Within coarse-colour-group pairwise separation.
    groups: dict[str, list[tuple]] = {}
    for e, c in rows:
        if c is None:
            continue
        groups.setdefault(_name_hue(c["h"]), []).append((e, c))

    print("\nWithin-colour pairwise separation (which space/dimension splits them?):")
    any_pair = False
    for colour, members in groups.items():
        if len(members) < 2:
            continue
        any_pair = True
        print(f"\n  [{colour}] {len(members)} rows:")
        for (e1, c1), (e2, c2) in combinations(members, 2):
            print(
                f"    {_label(e1)[:20]:20} vs {_label(e2)[:20]:20}  "
                f"HSVd={_hsv_dist(c1, c2):5.1f}  Labd={_lab_dist(c1, c2):5.1f}  "
                f"| dH={_hue_delta(c1['h'], c2['h']):4.0f} dS={c1['s']-c2['s']:+4.0f} "
                f"dV={c1['v']-c2['v']:+4.0f} dL={c1['L']-c2['L']:+4.0f} "
                f"da={c1['a']-c2['a']:+4.0f} db={c1['b']-c2['b']:+4.0f}"
            )
    if not any_pair:
        print("  (no same-colour groups with >1 row)")


def main() -> None:
    for name in IMAGES:
        run(name)
    print("\n" + "=" * 78)
    print("Read: within each colour group, is HSVd or Labd the larger separator, "
          "and which\nper-dimension delta (dH/dS/dV or da/db) carries it? "
          "That decides the anchor space.")


if __name__ == "__main__":
    main()
