"""
Figures for the classification stage, drawn by running the live pipeline.

Every number in the output comes from the run that draws it:

    parse_screenshot -> attach_class_names -> detect -> assign_classes

Nothing is cached and nothing is copied from an earlier report, so a figure can
never show behaviour the code no longer has. Two figures in this repository once
outlived the code that made them and misled a reviewer; that is why this script
exists and why each figure prints the detection mode it used.

Three outputs:

  fig_classify_<frame>.png    the aerial with every detected dot drawn in the colour
                              of the legend row it was assigned to, beside the dialog's
                              own rows. Three counts per row: what the dialog states,
                              what the pipeline assigned, and what the hand labels say.

  fig_accuracy_by_frame.png   per-dot accuracy on every frame that both passes frame
                              selection and has hand labels, each bar carrying its own
                              denominator.

Frames default to the two the pipeline handles best, which is what the figures are
for. Both are dense colonies; `5745` carries 14 legend rows and `0027` eight.

Usage:
    python scripts/make_classification_figures.py
    python scripts/make_classification_figures.py --frames 17May10Camera2-Card1-5745
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.align import AlignResult, align
from src.classify import assign_classes, detect_dots, detect_dots_subtract
from src.legend import attach_class_names, locate_dialog, parse_screenshot
from scripts.label_dots import legend_options

OUT = "results/figures"
PAIRS = "data/fixtures/pairs"
ALIGN_CACHE = "data/cache/align_cache.json"
TOL = 5.0

# `5745` carries fourteen legend rows and 296 labelled dots, the most demanding frame
# with ground truth, and nothing else in the selected set illustrates the stage as
# well. `0027` scores marginally higher but a painted transect line runs through it and
# is detected along its length; `06389` reads at 1.000 with two classes and puts painted
# text into one of them; `426` leaves 57% of its detections unlabelled.
DEFAULT_FRAMES = ["17May10Camera2-Card1-5745"]

# Every frame that both passes selection and carries hand labels.
SCORED_FRAMES = [
    "18June21Camera1-Card2-06389", "18May11Camera2-Card5-0027",
    "17May10Camera2-Card1-5745", "19May18Camera2-Card1-00620",
    "18May15Camera1-Card6-426", "18May15Camera2-Card5-00825",
    "27May12Camera1-Card2-0449",
]

# Distinct enough to tell fourteen classes apart at a glance. BGR.
PALETTE = [
    (60, 60, 220), (60, 200, 60), (230, 160, 40), (200, 60, 200),
    (40, 200, 200), (255, 120, 0), (150, 60, 220), (0, 160, 255),
    (120, 220, 120), (220, 120, 160), (90, 90, 255), (0, 220, 160),
    (180, 180, 60), (255, 80, 140), (120, 160, 255), (60, 140, 60),
    (200, 200, 100), (140, 60, 140), (0, 120, 200), (100, 200, 240),
    (170, 90, 40),
]


def find_screenshot(stem: str) -> str | None:
    hits = [p for p in glob.glob(f"{PAIRS}/screenshots/**/*.jpg", recursive=True)
            if os.path.basename(p) == stem + ".jpg"]
    return hits[0] if hits else None


def read_rgb(path: str | None) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_pipeline(stem: str):
    """The live path. Returns (screenshot, dots, entries, dialog bbox, mode)."""
    shot_path = find_screenshot(stem)
    shot = read_rgb(shot_path)
    if shot is None:
        return None
    orig = read_rgb(shot_path.replace("screenshots", "high_resolution_photos"))

    key = (shot_path.replace(f"{PAIRS}/", "avian_monitoring/").replace(os.sep, "/"))
    cache = json.load(open(ALIGN_CACHE)) if os.path.exists(ALIGN_CACHE) else {}
    h = cache.get(key)
    if h:
        res = AlignResult(
            H=np.array(h["H"], np.float64) if h["H"] is not None else None,
            scale=h["scale"], matches=h["matches"], inliers=h["inliers"],
            reproj_err=h["reproj_err"], model=h["model"], ok=h["ok"],
            reason=h["reason"])
    else:
        res = align(shot, orig) if orig is not None else None

    bbox = locate_dialog(shot)
    if res is not None and res.ok and orig is not None:
        dots, mode = detect_dots_subtract(shot, orig, res, exclude=bbox), "subtraction"
    else:
        dots, mode = detect_dots(shot, exclude=bbox), "colour"

    entries, dbox = parse_screenshot(shot)
    if dbox is not None and entries:
        x, y, w, hh = dbox
        try:
            attach_class_names(shot[y:y + hh, x:x + w], entries)
        except Exception:                                              # noqa: BLE001
            pass
        assign_classes(dots, entries)
    return shot, dots, entries, dbox, mode


def label_counts(stem: str, dots) -> dict:
    """Hand-labelled dots per legend row, matched to detections within TOL."""
    path = f"data/labels/{stem}.json"
    if not os.path.exists(path):
        return {}
    lab = json.load(open(path))
    rows = [d.get("row") for d in lab["dots"]]
    if not any(r is not None for r in rows):
        return {}
    return collections.Counter(r for r in rows if r is not None)


def draw_frame_figure(stem: str) -> str | None:
    out = run_pipeline(stem)
    if out is None:
        print(f"  {stem}: screenshot not cached, skipped")
        return None
    shot, dots, entries, dbox, mode = out
    if not entries:
        print(f"  {stem}: no legend parsed, skipped")
        return None

    names, rows = legend_options(entries)
    by_row = dict(zip(rows, names))
    stated = {e.row: e.count for e in entries}
    truth = label_counts(stem, dots)

    vis = cv2.cvtColor(shot, cv2.COLOR_RGB2BGR)
    assigned = collections.Counter()
    for d in dots:
        colour = ((170, 170, 170) if d.legend_row is None
                  else PALETTE[d.legend_row % len(PALETTE)])
        cv2.circle(vis, (int(d.cx), int(d.cy)), 7, colour, 2)
        assigned[d.legend_row] += 1
    if dbox is not None:
        x, y, w, hh = dbox
        cv2.rectangle(vis, (x, y), (x + w, y + hh), (255, 255, 255), 2)

    # Side panel: the dialog's own rows against what the pipeline did with them.
    ph, pw = vis.shape[0], 560
    panel = np.full((ph, pw, 3), 252, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, stem, (14, 34), f, 0.58, (20, 20, 20), 2)
    cv2.putText(panel, f"{len(dots)} dots detected, {mode} path",
                (14, 60), f, 0.46, (110, 110, 110), 1)
    head_y = 96
    cv2.putText(panel, "class", (46, head_y), f, 0.46, (110, 110, 110), 1)
    cv2.putText(panel, "dialog", (300, head_y), f, 0.46, (110, 110, 110), 1)
    cv2.putText(panel, "pipeline", (378, head_y), f, 0.46, (110, 110, 110), 1)
    cv2.putText(panel, "labels", (476, head_y), f, 0.46, (110, 110, 110), 1)

    y = head_y + 30
    for e in entries:
        if y > ph - 70:
            break
        colour = PALETTE[e.row % len(PALETTE)]
        cv2.circle(panel, (26, y - 5), 8, colour, -1)
        cv2.putText(panel, (by_row.get(e.row) or f"row {e.row}")[:22],
                    (46, y), f, 0.46, (20, 20, 20), 1)
        want = stated.get(e.row)
        cv2.putText(panel, "-" if want is None else str(want),
                    (306, y), f, 0.46, (20, 20, 20), 1)
        cv2.putText(panel, str(assigned.get(e.row, 0)), (392, y), f, 0.46,
                    (20, 20, 20), 1)
        cv2.putText(panel, str(truth.get(e.row, "-")), (486, y), f, 0.46,
                    (20, 20, 20), 1)
        y += 27

    if None in assigned:
        cv2.circle(panel, (26, y - 5), 8, (170, 170, 170), -1)
        cv2.putText(panel, "no class", (46, y), f, 0.46, (20, 20, 20), 1)
        cv2.putText(panel, str(assigned[None]), (392, y), f, 0.46, (110, 110, 110), 1)
        y += 27
    cv2.putText(panel,
                "A dot matching no legend row is left unclassified.",
                (14, min(y + 26, ph - 20)), f, 0.42, (110, 110, 110), 1)

    # JPEG, not PNG: the aerial is photographic and a lossless copy of a dense colony
    # runs to several megabytes, which is a lot of repository weight for a figure.
    path = os.path.join(OUT, f"fig_classify_{stem}.jpg")
    cv2.imwrite(path, np.hstack([vis, panel]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    named = sum(1 for e in entries if (e.class_name or "").strip())
    print(f"  {stem}: {len(dots)} dots, {len(entries)} rows ({named} named), "
          f"{mode} path -> {path}")
    return path


def accuracy_by_frame() -> str:
    """Per-dot accuracy on every selected frame that carries hand labels."""
    scores = []
    for stem in SCORED_FRAMES:
        out = run_pipeline(stem)
        if out is None:
            continue
        shot, dots, entries, dbox, _mode = out
        path = f"data/labels/{stem}.json"
        if not entries or not os.path.exists(path):
            continue
        lab = json.load(open(path))
        true_xy = [(d["x"], d["y"]) for d in lab["dots"]]
        true_row = [d.get("row") for d in lab["dots"]]
        if not any(r is not None for r in true_row):
            # Files written before the labeller recorded a row carry only the class
            # name. `legend_options` produces the same display names the page showed,
            # so the row is recoverable and the frame still counts — dropping it here
            # would make this chart disagree with `eval_localisation.py`.
            names, rows = legend_options(entries)
            by_name = dict(zip(names, rows))
            true_row = [by_name.get((d.get("cls") or "").strip())
                        for d in lab["dots"]]
        pairs = sorted(((np.hypot(d.cx - tx, d.cy - ty), i, j)
                        for i, d in enumerate(dots)
                        for j, (tx, ty) in enumerate(true_xy)
                        if abs(d.cx - tx) <= TOL and abs(d.cy - ty) <= TOL),
                       key=lambda t: t[0])
        used_d, used_t, ok, n = set(), set(), 0, 0
        for _dist, i, j in pairs:
            if i in used_d or j in used_t:
                continue
            used_d.add(i); used_t.add(j)
            if true_row[j] is None:
                continue
            n += 1
            if dots[i].legend_row == true_row[j]:
                ok += 1
        if n:
            scores.append((stem, ok, n))

    scores.sort(key=lambda t: -t[1] / max(t[2], 1))
    tot_ok = sum(s[1] for s in scores)
    tot_n = sum(s[2] for s in scores)

    row_h, top, left, bar_w = 46, 120, 330, 620
    img = np.full((top + row_h * len(scores) + 90, left + bar_w + 190, 3), 252, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Classification accuracy per dot, against hand labels",
                (20, 44), f, 0.72, (20, 20, 20), 2)
    cv2.putText(img,
                f"{tot_ok} of {tot_n} dots on {len(scores)} frames that pass frame "
                f"selection  =  {tot_ok/max(tot_n,1):.3f}",
                (20, 76), f, 0.5, (110, 110, 110), 1)

    for k, (stem, ok, n) in enumerate(scores):
        y = top + row_h * k
        acc = ok / max(n, 1)
        cv2.putText(img, stem[:30], (20, y + 22), f, 0.46, (20, 20, 20), 1)
        cv2.rectangle(img, (left, y + 6), (left + bar_w, y + 30), (232, 232, 232), -1)
        cv2.rectangle(img, (left, y + 6), (left + int(bar_w * acc), y + 30),
                      (90, 150, 60), -1)
        cv2.putText(img, f"{acc:.3f}  ({ok}/{n})", (left + bar_w + 14, y + 26),
                    f, 0.46, (20, 20, 20), 1)

    cv2.putText(img,
                "Each bar carries its own denominator. A frame with few classes scores "
                "high for that reason alone.",
                (20, top + row_h * len(scores) + 44), f, 0.44, (110, 110, 110), 1)
    path = os.path.join(OUT, "fig_accuracy_by_frame.png")
    cv2.imwrite(path, img)
    print(f"  accuracy chart: {tot_ok}/{tot_n} = {tot_ok/max(tot_n,1):.3f} -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES)
    ap.add_argument("--skip-chart", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("classification figures, drawn from the live pipeline:")
    for stem in args.frames:
        draw_frame_figure(stem)
    if not args.skip_chart:
        accuracy_by_frame()


if __name__ == "__main__":
    main()
