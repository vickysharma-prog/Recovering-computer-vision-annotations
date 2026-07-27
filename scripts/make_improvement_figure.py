"""
What changed, in two panels: detection accuracy and classification accuracy.

Both panels compare the same two configurations on the same images, so the
only difference is the code being measured.

Detection  — dots found divided by dots truly present, by density band, over
             the 60 benchmark pairs. 1.0 is exact; higher means over-detection.
             Old is colour thresholds, new is subtract-the-clean-original.
Classification — per-class assigned counts against the counts read from the
             dialog, averaged over the classes each legend lists. Old runs the
             previous matching (global colour bins, no background removal,
             binary-mask cosine, shape-name boost on) via the env toggles in
             classify.py; new runs the current default. Detection is held
             constant at the colour path for both, so the panel isolates
             matching.

Usage:
    python scripts/make_improvement_figure.py

Output: results/report_fig/fig_improvement.jpg
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "report_fig"

# Categorical slots 1 and 2 of the reference palette. Validated for CVD
# separation against the light surface (worst adjacent pair dE 24.7 protan).
C_OLD = "#2a78d6"
C_NEW = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

# Detection: median detected/true by band, over the 60 benchmark pairs.
# Recomputed from results/eval_detection.csv.
DET = [("sparse\n5-50 dots", 63.51, 2.13),
       ("medium\n51-300", 9.15, 1.24),
       ("dense\n301+", 3.56, 1.14)]

# Classification: mean per-class count agreement by band, over the 41 cached
# frames whose legend parses to at least two classes. Overall 0.263 -> 0.357;
# 27 frames improved, 14 got worse.
CLS = [("sparse\n7 frames", 0.223, 0.485),
       ("medium\n15 frames", 0.273, 0.352),
       ("dense\n13 frames", 0.266, 0.382)]
CLS_OVERALL = (0.263, 0.357)
CLS_SPLIT = (27, 14)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def build() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9),
                                   facecolor=SURFACE)
    h = 0.34            # bar height; the 0.02 gap is the surface spacer
    for ax in (ax1, ax2):
        _style(ax)

    # ── Detection ───────────────────────────────────────────────────────
    y = np.arange(len(DET))
    old = [d[1] for d in DET]
    new = [d[2] for d in DET]
    ax1.barh(y + h / 2 + 0.01, old, h, color=C_OLD, label="colour thresholds")
    ax1.barh(y - h / 2 - 0.01, new, h, color=C_NEW, label="subtraction")
    ax1.axvline(1.0, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax1.text(1.06, len(DET) - 0.62, "1.0 = exact", fontsize=8, color=INK_2)
    for i, (o, n) in enumerate(zip(old, new)):
        ax1.text(o * 1.12, i + h / 2 + 0.01, f"{o:.2f}x", va="center",
                 fontsize=9, color=INK_2)
        ax1.text(n * 1.12, i - h / 2 - 0.01, f"{n:.2f}x", va="center",
                 fontsize=9, color=INK, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlim(0.8, 200)
    ax1.set_yticks(y, [d[0] for d in DET])
    ax1.set_xticks([1, 10, 100], ["1x", "10x", "100x"])
    ax1.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("Detection — dots found / dots present\n"
                  "60 benchmark pairs, median per band. Closer to 1.0 is better.",
                  fontsize=10, color=INK, loc="left", pad=10)
    ax1.legend(frameon=False, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, labelcolor=INK_2)

    # ── Classification ──────────────────────────────────────────────────
    y = np.arange(len(CLS))
    old = [c[1] for c in CLS]
    new = [c[2] for c in CLS]
    ax2.barh(y + h / 2 + 0.01, old, h, color=C_OLD, label="previous matching")
    ax2.barh(y - h / 2 - 0.01, new, h, color=C_NEW, label="current matching")
    for i, (o, n) in enumerate(zip(old, new)):
        ax2.text(o + 0.012, i + h / 2 + 0.01, f"{o:.2f}", va="center",
                 fontsize=9, color=INK_2)
        better = n > o
        ax2.text(n + 0.012, i - h / 2 - 0.01,
                 f"{n:.2f}" + ("" if better else "  (down)"), va="center",
                 fontsize=9, color=INK, fontweight="bold")
    ax2.set_xlim(0, 0.62)
    ax2.set_yticks(y, [c[0] for c in CLS])
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    mo, mn = CLS_OVERALL
    up, down = CLS_SPLIT
    ax2.set_title("Classification — per-class count agreement (higher is better)\n"
                  f"41 frames, same detection, matching only. "
                  f"Overall {mo:.2f} to {mn:.2f}; {up} frames up, {down} down.",
                  fontsize=10, color=INK, loc="left", pad=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, labelcolor=INK_2)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_improvement.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE,
                pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)  "
          f"classification mean {mo:.3f} -> {mn:.3f}")


if __name__ == "__main__":
    build()
