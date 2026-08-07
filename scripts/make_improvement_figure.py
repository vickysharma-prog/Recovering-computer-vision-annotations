"""
Why some frames are usable and others are not, in two panels.

Both read `results/eval_localisation.csv` at run time, so the figure always shows
what the pipeline currently does. Nothing here is a stored constant: an earlier
version of this script hard-coded its numbers, went stale, and a reviewer read
behaviour that had already been replaced.

Left   How many dots a frame reports, against how many of them are real. Correct
       detections cannot outnumber the dots on the image, so precision is capped
       at 1/ratio. That ceiling is drawn; every frame sits under it by
       construction, and the useful question is which frames sit close to it.
Right  Precision and recall per frame, ordered the same way. Recall holds across
       all of them; precision is what separates.

Usage:
    python scripts/make_improvement_figure.py

Output: results/report_fig/fig_improvement.jpg
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "report_fig"
EVAL = ROOT / "results" / "eval_localisation.csv"

# Categorical slots 1 and 2 of the reference palette. Checked with the palette
# validator against this surface: worst adjacent pair dE 24.7 protan, 33.6 normal,
# both above the 8 and 15 floors.
C_KEEP = "#2a78d6"
C_DROP = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

# A frame is kept when it reports no more than twice the dots present. The bound
# puts a ceiling of 0.5 on anything beyond that, so there is nothing to recover
# downstream.
KEEP_MAX_RATIO = 2.0


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def load():
    rows = []
    with open(EVAL, newline="") as f:
        for r in csv.DictReader(f):
            survey = float(r["survey"] or 0)
            if survey <= 0:
                continue
            rows.append(dict(
                name=r["frame"].replace(".jpg", ""),
                short=r["frame"].split("-")[-1].replace(".jpg", ""),
                band=r["band"],
                ratio=int(r["detected"]) / survey,
                precision=float(r["precision"]),
                recall=float(r["recall"]),
                labels=int(r["labels"]),
            ))
    rows.sort(key=lambda d: d["ratio"])
    return rows


def build() -> None:
    rows = load()
    keep = [d for d in rows if d["ratio"] <= KEEP_MAX_RATIO]
    drop = [d for d in rows if d["ratio"] > KEEP_MAX_RATIO]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0), facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)

    # ── Left: the ceiling, and where frames sit under it ────────────────
    x = np.logspace(np.log10(0.4), np.log10(20), 200)
    ax1.plot(x, np.minimum(1.0, 1.0 / x), color=INK_2, lw=2.0,
             ls=(0, (5, 3)), zorder=1,
             label="ceiling: precision cannot exceed 1 / ratio")
    ax1.axvspan(0.4, KEEP_MAX_RATIO, color=C_KEEP, alpha=0.07, zorder=0)

    for grp, colour, lab in ((keep, C_KEEP, "kept"), (drop, C_DROP, "dropped")):
        if not grp:
            continue
        ax1.scatter([d["ratio"] for d in grp], [d["precision"] for d in grp],
                    s=110, color=colour, edgecolor=SURFACE, linewidth=2.0,
                    zorder=3, label=f"{lab}  ({len(grp)} frames)")

    # Label the kept frames and the single worst; more than that and the panel
    # stops being readable. Labels hang directly below their point, because every
    # point lies on or under the ceiling curve, so the space below is always free
    # and a side-placed label runs into either the curve or the axis.
    # The kept frames cluster in a small corner, so their labels are placed
    # individually rather than by one rule: the ceiling falls steeply through
    # this region and the points sit close enough to catch each other's text.
    # Lowest ratio goes above its point, the rest hang below or to the left.
    for i, d in enumerate(keep):
        if i == 0:                       # lowest ratio: clear space above
            off, va, ha = (0, 12), "bottom", "center"
        elif d["precision"] > 0.9:        # top of the panel: go left
            off, va, ha = (-10, -5), "top", "right"
        else:
            off, va, ha = (0, -11), "top", "center"
        ax1.annotate(f"{d['short']}\n{d['labels']} dots",
                     (d["ratio"], d["precision"]), textcoords="offset points",
                     xytext=off, fontsize=8, color=INK_2, va=va, ha=ha)
    if drop:
        d = drop[-1]
        ax1.annotate(f"{d['short']}\n{d['labels']} dots",
                     (d["ratio"], d["precision"]), textcoords="offset points",
                     xytext=(0, -11), fontsize=8, color=INK_2,
                     va="top", ha="center")

    ax1.set_xscale("log")
    ax1.set_xlim(0.4, 20)
    ax1.set_ylim(-0.20, 1.08)   # room for a label hanging below the lowest point
    ax1.set_xticks([0.5, 1, 2, 5, 10, 20],
                   ["0.5x", "1x", "2x", "5x", "10x", "20x"])
    ax1.set_xlabel("dots reported / dots present", fontsize=9, color=INK_2)
    ax1.set_ylabel("measured precision", fontsize=9, color=INK_2)
    ax1.grid(color=GRID, lw=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("Which frames are worth using\n"
                  "Reported count is known without labels, so the ceiling is too.",
                  fontsize=10, color=INK, loc="left", pad=10)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right",
               labelcolor=INK_2)

    # ── Right: precision and recall per frame, same order ───────────────
    y = np.arange(len(rows))
    h = 0.34
    ax2.barh(y + h / 2 + 0.01, [d["recall"] for d in rows], h,
             color=INK_2, alpha=0.45, label="recall")
    ax2.barh(y - h / 2 - 0.01, [d["precision"] for d in rows], h,
             color=[C_KEEP if d["ratio"] <= KEEP_MAX_RATIO else C_DROP
                    for d in rows])
    for i, d in enumerate(rows):
        ax2.text(d["precision"] + 0.015, i - h / 2 - 0.01,
                 f"{d['precision']:.2f}", va="center", fontsize=8.5,
                 color=INK, fontweight="bold")

    ax2.set_yticks(y, [f"{d['short']}  ({d['ratio']:.2f}x)" for d in rows],
                   fontsize=8.5)
    ax2.set_xlim(0, 1.15)
    ax2.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("Recall holds everywhere; precision is what splits\n"
                  f"{sum(d['labels'] for d in rows)} hand-labelled dots, "
                  f"{len(rows)} frames, ordered by ratio.",
                  fontsize=10, color=INK, loc="left", pad=10)
    # Precision bars take the frame's own kept/dropped colour, so the legend has
    # to name both rather than show one swatch that half the bars contradict.
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=INK_2, alpha=0.45, label="recall"),
                        Patch(facecolor=C_KEEP, label="precision, kept"),
                        Patch(facecolor=C_DROP, label="precision, dropped")],
               frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_improvement.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE,
                pil_kwargs={"quality": 88})
    plt.close(fig)

    kp = [d["precision"] for d in keep]
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    print(f"  {len(rows)} frames, {sum(d['labels'] for d in rows)} labelled dots")
    if kp:
        print(f"  kept {len(keep)}: precision {min(kp):.2f}-{max(kp):.2f}")
    if drop:
        dp = [d["precision"] for d in drop]
        print(f"  dropped {len(drop)}: precision {min(dp):.2f}-{max(dp):.2f}")


if __name__ == "__main__":
    build()
