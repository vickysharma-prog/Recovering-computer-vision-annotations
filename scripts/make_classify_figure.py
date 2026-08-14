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


def _verdicts(shot_path: Path | None, dots, entries, tol: float = 5.0) -> dict:
    """
    For each detection, whether the hand label at that point agrees with the class it
    was given: `right`, `wrong`, or absent when no label sits there.

    Without this the figure shows only what the pipeline decided, and a row can look
    convincing while being wrong. On `5745` the two rows holding the fewest dots each
    match the dialog's stated count exactly and are wrong on every dot — a reader
    seeing counts agree would conclude the opposite.

    Returns `{id(dot): verdict}`, empty when the frame has no labels.
    """
    if shot_path is None:
        return {}
    path = ROOT / "data" / "labels" / (shot_path.stem + ".json")
    if not path.exists():
        return {}
    import json
    lab = json.load(open(path))
    xy = [(d["x"], d["y"]) for d in lab["dots"]]
    rows = [d.get("row") for d in lab["dots"]]
    if not any(r is not None for r in rows):
        return {}

    pairs = sorted(((np.hypot(d.cx - tx, d.cy - ty), i, j)
                    for i, d in enumerate(dots)
                    for j, (tx, ty) in enumerate(xy)
                    if abs(d.cx - tx) <= tol and abs(d.cy - ty) <= tol),
                   key=lambda t: t[0])
    used_d, used_t, out = set(), set(), {}
    for _dist, i, j in pairs:
        if i in used_d or j in used_t:
            continue
        used_d.add(i); used_t.add(j)
        if rows[j] is None:
            continue
        out[id(dots[i])] = "right" if dots[i].legend_row == rows[j] else "wrong"
    return out


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
    verdicts = _verdicts(shot_path, dots, entries)

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
    # The title and the column headings need a fixed band whatever the row count.
    # Without it a frame with only two populated classes draws the title straight
    # over the headings. The band grows with the title, which gains a line once
    # hand labels are available to score against.
    header_in = 1.5 + (0.32 if verdicts else 0.0)
    fig, axes = plt.subplots(len(order), n_cols,
                             figsize=(1.28 * n_cols,
                                      1.28 * len(order) + header_in))
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
            colour = {"right": "#2e9e4f", "wrong": "#d2342b"}.get(
                verdicts.get(id(pick[c])), "white")
            # Tight enough to pick out one marker. A wider ring covers its
            # neighbours too, and in a colony that leaves the reader unable to
            # tell which dot the verdict belongs to.
            ax.add_patch(plt.Circle((w / 2, h / 2), min(h, w) * 0.18,
                                    fill=False, color=colour, lw=1.5))

    axes[0, 0].set_title("dialog\nmarker", fontsize=8)
    axes[0, 1].set_title("template\n(24x24)", fontsize=8)
    checked = any(v in ("right", "wrong") for v in verdicts.values())
    axes[0, 2].set_title(
        "sample aerial patches assigned to this class   "
        + ("green ring = a hand label at that point agrees, red = it disagrees, "
           "white = no hand label there" if checked
           else "white ring = matched point"),
        fontsize=8, loc="left")

    total = sum(len(v) for v in by_entry.values())
    third = ""
    if checked:
        right = sum(1 for d in dots if verdicts.get(id(d)) == "right")
        wrong = sum(1 for d in dots if verdicts.get(id(d)) == "wrong")
        blank = len(dots) - right - wrong
        # Hand labelling is not exhaustive, and detection also finds things that are
        # not markers, so an unringed patch is unverified rather than wrong. Stating
        # the split stops a reader reading every white ring as an error.
        third = (f"\nover all {len(dots)} detections on this frame: {right} agree with "
                 f"a hand label, {wrong} disagree, {blank} have no label "
                 f"({blank / max(len(dots), 1):.0%})")
    fig.suptitle(
        f"Classification — current pipeline   ·   {shot_path.name}\n"
        f"Lab colour anchoring + hue background removal + NCC shape match, no shape-name boost"
        f"   ·   detection: {mode}   ·   {total} dots assigned across {len(by_entry)} classes"
        + third,
        fontsize=10.5)
    # Leave the reserved band free instead of letting tight_layout reclaim it.
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig.get_figheight()))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"fig_classify_{key}.jpg"
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 78})
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)  mode={mode}  "
          f"rows={len(order)}  dots={total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None,
                    help="output name. Defaults to the frame given by --pair, so the "
                         "file is named after what it shows; falls back to the old "
                         "four-study-image key when no pair is given.")
    ap.add_argument("--pair", default=None,
                    help="benchmark frame name; enables the subtraction path")
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()
    key = args.key or (os.path.splitext(args.pair)[0] if args.pair
                       else "D_raccoon_2011")
    build(key, args.pair, args.rows, args.samples)
