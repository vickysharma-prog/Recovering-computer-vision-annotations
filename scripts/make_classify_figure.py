"""
Classification figure: dialog marker, the template cut from it, and the aerial
patches the current matcher assigns to that class.

Layout matches the earlier figure so rows can be compared directly: one row per
class, left is the dialog marker crop, then the 24x24 derived template, then
sample aerial patches with a white ring on the matched point.

Runs the live path, so the figure always shows current behaviour:
    parse_screenshot -> attach_class_names -> detect -> assign_classes

Detection is the subtraction path when a clean original is available for the
frame, and the colour path otherwise. The mode is printed in the title so the
figure never overstates what produced it.

Sample patches are drawn with a fixed seed from the full assigned set, not
picked by score, so a row shows typical matches rather than the best ones.

Usage:
    python scripts/make_classify_figure.py                      # image D, colour path
    python scripts/make_classify_figure.py --pair 18June11Camera2-Card7-586.jpg
    python scripts/make_classify_figure.py --rows 8 --samples 6

Output: results/report_fig/fig_classify_<key>.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legend import parse_screenshot, attach_class_names
from src.classify import detect_dots, detect_dots_subtract, assign_classes

ROOT = Path(__file__).parent.parent
SCREENSHOTS = ROOT / "data" / "fixtures" / "screenshots"
PAIRS = ROOT / "data" / "fixtures" / "pairs"
OUT = ROOT / "results" / "report_fig"

_RNG = np.random.default_rng(0)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _find_pair(name: str) -> tuple[Path, Path] | None:
    """Locate a screenshot / clean-original pair by file name."""
    shot = next(iter(PAIRS.glob(f"screenshots/**/{name}")), None)
    orig = next(iter(PAIRS.glob(f"high_resolution_photos/**/{name}")), None)
    return (shot, orig) if shot and orig else None


def _marker_crop(rgb: np.ndarray, bbox, e, pad: int = 9) -> np.ndarray:
    """The dialog marker as it appears in the screenshot."""
    bx, by = bbox[0], bbox[1]
    cx, cy = int(bx + e.cx), int(by + e.cy)
    y0, y1 = max(0, cy - pad), min(rgb.shape[0], cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(rgb.shape[1], cx + pad + 1)
    return rgb[y0:y1, x0:x1]


def _patch(rgb: np.ndarray, d, half: int = 22) -> np.ndarray | None:
    y0, y1 = int(d.cy) - half, int(d.cy) + half
    x0, x1 = int(d.cx) - half, int(d.cx) + half
    if y0 < 0 or x0 < 0 or y1 > rgb.shape[0] or x1 > rgb.shape[1]:
        return None
    return rgb[y0:y1, x0:x1]


def build(key: str, pair: str | None, n_rows: int, n_samples: int) -> None:
    if pair:
        found = _find_pair(pair)
        if not found:
            raise SystemExit(f"no pair found for {pair}")
        shot_path, orig_path = found
        rgb = _read_rgb(shot_path)
        original = _read_rgb(orig_path)
        mode = "subtraction"
    else:
        shot_path = next(iter(SCREENSHOTS.glob(f"{key}.*")), None)
        if shot_path is None:
            raise SystemExit(f"no screenshot for {key}")
        rgb = _read_rgb(shot_path)
        original = None
        mode = "colour"

    entries, bbox = parse_screenshot(rgb)
    dialog = rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    attach_class_names(dialog, entries)

    if original is not None:
        from src.align import align
        res = align(rgb, original)
        if res.ok:
            dots = detect_dots_subtract(rgb, original, res, exclude=bbox)
        else:
            dots, mode = detect_dots(rgb, exclude=bbox), "colour (alignment refused)"
    else:
        dots = detect_dots(rgb, exclude=bbox)

    assign_classes(dots, entries)

    # Group by assigned class, largest first, same ordering the old figure used.
    by_entry: dict[int, list] = {}
    for d in dots:
        if getattr(d, "entry_row", None) is None and not d.class_name:
            continue
        for i, e in enumerate(entries):
            if e.class_name and d.class_name == e.class_name:
                by_entry.setdefault(i, []).append(d)
                break
    order = sorted(by_entry, key=lambda i: -len(by_entry[i]))[:n_rows]
    if not order:
        raise SystemExit("no dots were assigned to any class")

    n_cols = 2 + n_samples
    fig, axes = plt.subplots(len(order), n_cols,
                             figsize=(1.28 * n_cols, 1.28 * len(order)))
    if len(order) == 1:
        axes = axes[None, :]

    for r, i in enumerate(order):
        e = entries[i]
        group = by_entry[i]
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.4); s.set_color("#bbbbbb")

        axes[r, 0].imshow(_marker_crop(rgb, bbox, e))
        axes[r, 1].imshow(e.template, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"{e.class_name}\n({e.color}/{e.shape}, n={len(group)})",
                              rotation=0, ha="right", va="center", fontsize=7.5,
                              labelpad=8)

        # Only dots far enough from the frame edge to crop a full patch, so a
        # row shows six real matches instead of blanks. Sampled with a fixed
        # seed rather than ranked by score: these are typical, not the best.
        pick = [d for d in group if _patch(rgb, d) is not None]
        if len(pick) > n_samples:
            idx = _RNG.choice(len(pick), n_samples, replace=False)
            pick = [pick[j] for j in sorted(idx)]
        for c in range(n_samples):
            ax = axes[r, 2 + c]
            p = _patch(rgb, pick[c]) if c < len(pick) else None
            if p is None or p.size == 0:
                ax.axis("off")
                continue
            ax.imshow(p)
            h, w = p.shape[:2]
            ax.add_patch(plt.Circle((w / 2, h / 2), min(h, w) * 0.30,
                                    fill=False, color="white", lw=1.1))

    axes[0, 0].set_title("dialog\nmarker", fontsize=8)
    axes[0, 1].set_title("template\n(24x24)", fontsize=8)
    axes[0, 2].set_title("sample aerial patches assigned to this class "
                         "(white ring = matched point)",
                         fontsize=8, loc="left")

    total = sum(len(v) for v in by_entry.values())
    fig.suptitle(
        f"Classification — current pipeline   ·   {shot_path.name}\n"
        f"Lab colour anchoring + hue background removal + NCC shape match, no shape-name boost"
        f"   ·   detection: {mode}   ·   {total} dots assigned across {len(by_entry)} classes",
        fontsize=10.5, y=1.005)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"fig_classify_{key}.jpg"
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 78})
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)  mode={mode}  "
          f"rows={len(order)}  dots={total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="D_raccoon_2011")
    ap.add_argument("--pair", default=None,
                    help="benchmark frame name; enables the subtraction path")
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()
    build(args.key, args.pair, args.rows, args.samples)
