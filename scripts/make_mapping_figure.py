"""
Mapping figure: the recovered dots, placed on the original photograph.

Every earlier stage works in the screenshot's pixel grid. This figure is the check
that `src/mapping.py` puts those dots where the birds actually are, at full
resolution, which is the only claim that matters before a dataset is built from them.

Two panels per frame:

    left    the whole original, with every mapped dot. The dots should fill the
            photograph. A missing `/ res.scale` in the transform would crowd them
            into one corner while still looking internally consistent.
    right   the densest window at full resolution, with the provisional export box
            drawn. Birds should be visible under the boxes.

Runs the live path — cached transform, `detect_dots_subtract`, `map_dots` — and prints
the detection mode used for each frame, because a figure that outlives the code that
produced it has already misled a reviewer here once.

Usage:
    python scripts/make_mapping_figure.py
    python scripts/make_mapping_figure.py --box 80

Output: results/figures/fig_mapping_to_original.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.align import AlignResult
from src.birdsize import box_from_size, frame_bird_size
from src.classify import detect_dots, detect_dots_subtract
from src.legend import locate_dialog
from src.mapping import map_dots

ROOT = Path(__file__).parent.parent
PAIRS = ROOT / "data" / "fixtures" / "pairs"
CACHE = ROOT / "data" / "cache" / "align_cache.json"
OUT = ROOT / "results" / "figures"

# One dense colony with hand labels, one dense without, one sparse. The sparse frame
# is here because it is the case a mapping bug would hide in: ten dots anywhere on a
# 5184x3456 photograph still look plausible.
FRAMES = ["17May10Camera2-Card1-5745",
          "18May11Camera2-Card5-0027",
          "18May15Camera2-Card5-00825"]

HALF = 350          # half-width of the zoom window, in original pixels


def _read(key: str):
    img = cv2.imread(str(PAIRS / key.replace("avian_monitoring/", "")))
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _result(hit: dict) -> AlignResult:
    return AlignResult(
        H=np.array(hit["H"], np.float64) if hit["H"] is not None else None,
        scale=hit["scale"], matches=hit["matches"], inliers=hit["inliers"],
        reproj_err=hit["reproj_err"], model=hit["model"], ok=hit["ok"],
        reason=hit["reason"])


def _densest(pts: np.ndarray, shape) -> tuple[int, int]:
    """Centre of the window holding the most dots, on a coarse grid."""
    h, w = shape[:2]
    best, bx, by = -1, w // 2, h // 2
    for cy in range(HALF, h - HALF, 100):
        for cx in range(HALF, w - HALF, 100):
            n = int(((np.abs(pts[:, 0] - cx) < HALF)
                     & (np.abs(pts[:, 1] - cy) < HALF)).sum())
            if n > best:
                best, bx, by = n, cx, cy
    return bx, by


def build(forced: int) -> None:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    fig, axes = plt.subplots(len(FRAMES), 2, figsize=(14, 6.2 * len(FRAMES)),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    total_dots = 0

    # Measured first, drawn second: a frame whose own birds cannot be measured
    # borrows the median of the ones that could, exactly as the export does. Drawing
    # as we go would have left that frame with no box at all.
    prepared = []
    for name in FRAMES:
        key = next((k for k in cache if name in k), None)
        if key is None:
            raise SystemExit(f"{name} is not in {CACHE}")
        res = _result(cache[key])
        shot = _read(key)
        orig = _read(key.replace("screenshots", "high_resolution_photos"))
        if shot is None or orig is None:
            raise SystemExit(f"{name}: screenshot or original not cached")

        try:
            bbox = locate_dialog(shot)
        except Exception:                                          # noqa: BLE001
            bbox = None
        if res.ok:
            dots, mode = detect_dots_subtract(shot, orig, res, exclude=bbox), "subtract"
        else:
            raise SystemExit(f"{name}: alignment rejected, nothing to map onto")

        mapped = [m for m in map_dots(dots, res, orig.shape) if m.in_bounds]
        total_dots += len(mapped)
        pts = np.array([(m.x, m.y) for m in mapped], dtype=np.float64)
        h, w = orig.shape[:2]

        # The box is this frame's own, measured the same way the export measures it,
        # so the figure cannot show a box the dataset does not contain.
        est = frame_bird_size(orig, [(m.x, m.y) for m in mapped])
        prepared.append((name, orig, res, mapped, pts, est, mode))

    measured = [box_from_size(p[5].median_px) for p in prepared if p[5].ok]
    fallback = forced or (int(np.median(measured)) if measured else 0)

    for r, (name, orig, res, mapped, pts, est, mode) in enumerate(prepared):
        h, w = orig.shape[:2]
        box = box_from_size(est.median_px) if est.ok else fallback
        note = "" if est.ok else "  (borrowed: this frame's birds are not measurable)"
        print(f"  {name:34s} mode={mode:9s} dots={len(mapped):5d} "
              f"orig={w}x{h} scale={res.scale:.4f} reproj={res.reproj_err:.2f}px "
              f"bird={est.median_px if est.ok else float('nan'):.1f}px box={box}px{note}")

        # ---- left: the whole photograph
        ax = axes[r, 0]
        ax.imshow(orig)
        ax.scatter(pts[:, 0], pts[:, 1], s=9, facecolors="none",
                   edgecolors="#ff2200", linewidths=0.7)
        cx, cy = _densest(pts, orig.shape)
        ax.add_patch(Rectangle((cx - HALF, cy - HALF), 2 * HALF, 2 * HALF,
                               fill=False, edgecolor="#00ccff", linewidth=1.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{len(mapped)} dots mapped onto {w}x{h} "
                     f"(scale {res.scale:.3f}, reprojection {res.reproj_err:.2f}px)",
                     fontsize=10)

        # ---- right: the densest window, full resolution
        x0, y0 = cx - HALF, cy - HALF
        ax = axes[r, 1]
        ax.imshow(orig[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF])
        inside = [m for m in mapped
                  if x0 <= m.x < x0 + 2 * HALF and y0 <= m.y < y0 + 2 * HALF]
        for m in inside:
            ax.add_patch(Rectangle((m.x - x0 - box / 2, m.y - y0 - box / 2), box, box,
                                   fill=False, edgecolor="#ffcc00", linewidth=0.8))
            ax.plot(m.x - x0, m.y - y0, marker="+", color="#ff2200",
                    markersize=5, markeredgewidth=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"full resolution, {len(inside)} dots  ·  box {box}x{box}px, "
                     f"measured from this frame's own birds", fontsize=10)

    fig.suptitle(
        "Recovered dots placed on the original photograph\n"
        "live path: cached alignment -> detect_dots_subtract -> map_dots -> birdsize"
        f"   ·   {total_dots} dots over {len(FRAMES)} frames",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_mapping_to_original.png"
    fig.savefig(out, dpi=95, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=0,
                    help="force a box side; the default measures each frame")
    build(ap.parse_args().box)
