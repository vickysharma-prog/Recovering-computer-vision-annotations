"""
The project's measured progress on one screen.

Comet's panels draw a line only for a metric with many points along a step
axis, which means training epochs. This project has seven dated milestones, so
Comet renders them as seven dots. This script draws the chart instead, and it
gets logged to Comet as an image.

Left  : detection error at each milestone, overall and on the sparse band,
        which was always the weakest.
Right : classification agreement before and after the matching rework, with
        the self-recovery number beside it so the gap is visible.

Numbers are transcribed from the dated entries in docs/progress_report.md.
Comet was set up after that work, so this history was recorded in the report
rather than captured live. Runs from here on are logged as they happen.

Usage:
    python scripts/make_timeline_figure.py

Output: results/report_fig/fig_timeline.jpg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "report_fig"

# Categorical slots 1 and 2 of the reference palette, validated for CVD
# separation against the light surface.
C_A = "#2a78d6"
C_B = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

# Milestones that carry a detection measurement. The three earlier ones
# (legend module, count-OCR, matching rework) predate the benchmark, so there
# is no comparable detection number to plot for them.
STEPS = [
    ("20 Jul\ncolour\nthresholds", 8.403, 63.51),
    ("20 Jul\nsubtract the\noriginal", 1.46, 6.07),
    ("24 Jul\nsaturation\nfloor", 1.244, 2.132),
    ("3 Aug\nwired into\nclassifier", 1.244, 2.132),
]

CLS = [("previous\nmatching", 0.263), ("current\nmatching", 0.357)]
SELF_RECOVERY = 0.795          # midpoint of D 76% and A 83%


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def build() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                   gridspec_kw={"width_ratios": [1.75, 1]},
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)

    # ── Detection over the project ──────────────────────────────────────
    x = np.arange(len(STEPS))
    med = [s[1] for s in STEPS]
    spa = [s[2] for s in STEPS]

    ax1.axhline(1.0, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax1.text(-0.28, 1.07, "1.0 = exact count", fontsize=8.5,
             color=INK_2, ha="left")

    ax1.plot(x, spa, "-o", color=C_A, lw=2.2, ms=7, label="sparse frames (the hardest)")
    ax1.plot(x, med, "-o", color=C_B, lw=2.2, ms=7, label="all frames (median)")

    for xi, (v, colour, dy) in enumerate(
            [(spa[i], C_A, 1.35) for i in range(len(spa))]):
        ax1.annotate(f"{v:.2f}x", (xi, v), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9,
                     color=C_A, fontweight="bold")
    for xi, v in enumerate(med):
        ax1.annotate(f"{v:.2f}x", (xi, v), textcoords="offset points",
                     xytext=(0, -17), ha="center", fontsize=9,
                     color=C_B, fontweight="bold")

    ax1.set_yscale("log")
    ax1.set_ylim(0.7, 160)
    ax1.set_xlim(-0.35, len(STEPS) - 0.55)
    ax1.set_xticks(x, [s[0] for s in STEPS])
    ax1.set_yticks([1, 10, 100], ["1x", "10x", "100x"])
    ax1.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("Detection: dots found / dots present\n"
                  "Every step is a measured change. Closer to 1.0 is better.",
                  fontsize=11, color=INK, loc="left", pad=12)
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK_2)

    # ── Classification ──────────────────────────────────────────────────
    xc = np.arange(len(CLS))
    vals = [c[1] for c in CLS]
    ax2.bar(xc, vals, 0.5, color=[C_A, C_B])
    for xi, v in zip(xc, vals):
        ax2.text(xi, v + 0.018, f"{v:.2f}", ha="center", fontsize=10,
                 color=INK, fontweight="bold")

    ax2.axhline(SELF_RECOVERY, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax2.text(len(CLS) - 0.5, SELF_RECOVERY + 0.02,
             "0.76-0.83 when the method is\ntested on its own templates",
             fontsize=8.5, color=INK_2, ha="right")

    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(-0.6, len(CLS) - 0.4)
    ax2.set_xticks(xc, [c[0] for c in CLS])
    ax2.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Classification: per-class count agreement\n"
                  "41 frames, same detection. Higher is better.",
                  fontsize=11, color=INK, loc="left", pad=12)

    fig.text(0.008, -0.02,
             "Detection is close to done; classification is the weak half. "
             "History transcribed from the dated entries in the progress "
             "report, since experiment tracking was set up afterwards.",
             fontsize=8.5, color=INK_2)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_timeline.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE,
                pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
