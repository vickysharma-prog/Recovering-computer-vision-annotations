"""
Box figure: the exported dataset drawn on the photographs, every frame.

This draws what `results/dataset/annotations_deepforest.csv` actually contains, rather
than recomputing anything, so it is a check on the deliverable and not on a copy of
it. If a box in this figure is wrong, the box in the dataset is wrong.

One panel per exported frame, so the whole set can be judged at once. Judging a box
size from three or four frames is how the flat 100px box survived as long as it did:
it looked defensible on the sparse frames and was four times too large on the dense
ones.

Each panel is the densest window of that frame at full resolution, with every
exported box drawn. The caption carries the frame's measured bird size and the box
derived from it, so a panel where the boxes look wrong can be traced to a number.

Usage:
    python scripts/make_box_figure.py
    python scripts/make_box_figure.py --window 260

Output: results/figures/fig_box_per_frame.png
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
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
PAIRS = ROOT / "data" / "fixtures" / "pairs"
DATASET = ROOT / "results" / "dataset" / "annotations_full.csv"
OUT = ROOT / "results" / "figures"


def densest(pts: np.ndarray, shape, half: int) -> tuple[int, int]:
    """Centre of the window holding the most dots, on a coarse grid."""
    h, w = shape[:2]
    best, bx, by = -1, w // 2, h // 2
    step = max(60, half // 2)
    for cy in range(half, max(half + 1, h - half), step):
        for cx in range(half, max(half + 1, w - half), step):
            n = int(((np.abs(pts[:, 0] - cx) < half)
                     & (np.abs(pts[:, 1] - cy) < half)).sum())
            if n > best:
                best, bx, by = n, cx, cy
    return bx, by


def build(window: int) -> None:
    if not DATASET.exists():
        raise SystemExit(f"{DATASET} not found. Run scripts/export_dataset.py first.")
    full = pd.read_csv(DATASET)
    e = full[full.exported]
    frames = sorted(e.frame.unique())
    half = window // 2

    cols = 5
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.35 * rows))
    axes = np.atleast_2d(axes)

    measured = 0
    for i, name in enumerate(frames):
        ax = axes[i // cols, i % cols]
        g = e[e.frame == name]
        img = cv2.imread(str(PAIRS / g.image_path.iloc[0]))
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pts = g[["x_orig", "y_orig"]].to_numpy()
        cx, cy = densest(pts, img.shape, half)
        x0, y0 = cx - half, cy - half
        ax.imshow(img[y0:y0 + window, x0:x0 + window])

        box = float(g.box_px.iloc[0])
        inside = 0
        for _j, r in g.iterrows():
            if not (x0 <= r.x_orig < x0 + window and y0 <= r.y_orig < y0 + window):
                continue
            inside += 1
            ax.add_patch(Rectangle((r.x_orig - x0 - box / 2, r.y_orig - y0 - box / 2),
                                   box, box, fill=False, edgecolor="#22dd22",
                                   linewidth=1.1))

        bird = g.bird_px.iloc[0]
        is_measured = bool(g.box_measured.iloc[0])
        measured += int(is_measured)
        note = (f"bird {bird:.1f}px -> box {box:.0f}px" if is_measured
                else f"not measurable, box {box:.0f}px borrowed")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name[:30]}\n{note}   ·   {inside} of {len(g)} dots shown",
                     fontsize=8.5, color="#111111" if is_measured else "#aa3300")

    for j in range(len(frames), rows * cols):
        axes[j // cols, j % cols].axis("off")

    boxes = e.groupby("frame").box_px.first()
    fig.suptitle(
        "Every exported frame carries its own box, measured from its own birds\n"
        f"{len(frames)} frames, {len(e)} boxes   ·   "
        f"box {boxes.min():.0f}-{boxes.max():.0f}px, median {boxes.median():.0f}px   "
        f"·   measured on {measured} of {len(frames)} frames   ·   "
        f"windows are {window}x{window}px at full resolution",
        fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_box_per_frame.png"
    fig.savefig(out, dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)  "
          f"{len(frames)} frames, box {boxes.min():.0f}-{boxes.max():.0f}px")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=300,
                    help="side of the full-resolution window shown per frame")
    build(ap.parse_args().window)
