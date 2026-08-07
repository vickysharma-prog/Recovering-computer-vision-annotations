"""
Build a click-to-label page per benchmark frame, so dot positions become data.

Why this exists: the survey gives a COUNT per image and never a coordinate. A
count cannot distinguish "found all 61 markers" from "found 40 real ones and
invented 21" from "found 61 in the wrong places" — all three score 61. So
detection precision/recall, placement error, and per-dot class accuracy are all
currently unmeasurable, and every remaining tuning decision (the 0.35/1.5/3.0
size band in `subtract.dot_candidates`, the choice of marker-size estimator) is
being made blind.

Labelling here is transcription, not judgement. The annotator who drew these dots
in 2010 already decided where the birds were; the marker is sitting in the pixels.
This only writes down where, in a form a script can check against.

Two guards against the bias that seeding introduces, since a labeller shown the
detector's output tends to confirm it rather than hunt for what it missed:
  * the page tracks a sweep grid and only credits a tile once it has been viewed
    at 2x or closer, which is the zoom at which a 4px marker is visible at all;
  * `--blind N` leaves N frames unseeded as a control. If the blind frames turn
    up materially more dots per image than the seeded ones, seeded recall is
    optimistic and has to be reported with that caveat.

Usage:
    python scripts/label_dots.py                 # 10 frames, 2 of them blind
    python scripts/label_dots.py --frames 20
    python scripts/label_dots.py --only 10June10Camera1-Card1-0076.jpg

Then open each file in results/labelling/*.html, click, press S to save, and put
the downloaded .json into data/labels/.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import align, AlignResult
from src.classify import detect_dots
from src.legend import locate_dialog, parse_screenshot, attach_class_names
from src.subtract import extract_annotations, dot_candidates

BENCHMARK = "data/cache/benchmark.csv"
PAIR_DIR = "data/fixtures/pairs"
ALIGN_CACHE = "data/cache/align_cache.json"
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeller.html")
OUT = "results/labelling"

# Frames whose behaviour a label set has to explain, kept in every selection.
# These are the ones driving the open decisions, so labelling a random 10 that
# excluded them would answer nothing.
PINNED = [
    "10June10Camera1-Card1-0076.jpg",    # worst over-detect, 14.3x, mangrove
    "18June21Camera1-Card2-06389.jpg",   # worst under-detect, 0.18x, band too high
    "14June21Camera1-Card1-238.jpg",     # heavy fusion: 21 blobs for 71 markers
    "18May11Camera2-Card5-0081.jpg",     # 51% of blobs fall below the size band
    "16May15Camera2-Card1-00097.jpg",    # ground-truth artifact: truth 450, 0 dots
]


def read(key: str):
    path = os.path.join(PAIR_DIR, key.replace("avian_monitoring/", ""))
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cached_align(key, shot, orig, cache):
    hit = cache.get(key)
    if hit is not None:
        return AlignResult(
            H=np.array(hit["H"], np.float64) if hit["H"] is not None else None,
            scale=hit["scale"], matches=hit["matches"], inliers=hit["inliers"],
            reproj_err=hit["reproj_err"], model=hit["model"], ok=hit["ok"],
            reason=hit["reason"])
    return align(shot, orig)


def seeds_for(shot, orig, res, bbox, entries):
    """Union of both detectors, so the labeller rejects more than they add.

    Deliberately the union and not the shipped path: a seed the detector should
    not have produced is cheap to delete, whereas a marker neither detector found
    is one the labeller has to notice unaided.

    Each seed also carries the pipeline's own class guess as `pred` — which the
    page never displays. Showing it would defeat the purpose: the known failure is
    two same-colour classes whose labels get swapped wholesale (WHIB site vs WHIB
    bird scored 0.548 against 0.540 on the same dot), and a labeller shown a
    confident wrong answer tends to accept it. Kept in the file so the eval can
    build a confusion matrix of human against pipeline afterwards, with the human
    label formed independently.
    """
    dots = []
    if res is not None and res.ok:
        try:
            from src.classify import detect_dots_subtract
            dots = list(detect_dots_subtract(shot, orig, res, exclude=bbox))
        except Exception:                                          # noqa: BLE001
            dots = []
    if not dots:
        # Colour only where subtraction could not run. Taking the union of both,
        # which the first version of this script did, inherits the worse of the
        # two: the colour path over-detects 184x on 0076, so the union produced
        # 1571 seeds against 9 real markers. Seeds are meant to save the labeller
        # work, and a seed set that large costs more to delete than to draw.
        try:
            dots = list(detect_dots(shot, exclude=bbox))
        except Exception:                                          # noqa: BLE001
            dots = []
    if bbox is not None:
        x, y, w, h = bbox
        dots = [d for d in dots if not (x <= d.cx <= x + w and y <= d.cy <= y + h)]

    if entries:
        try:
            # Mutates the dots in place, setting `class_name` on each, and returns
            # the same list — so there is nothing to zip against.
            from src.classify import assign_classes
            assign_classes(dots, entries)
        except Exception:                                          # noqa: BLE001
            pass

    # Merge near-duplicates: within ~4px is one marker.
    keep = []
    for d in dots:
        if all((d.cx - k.cx) ** 2 + (d.cy - k.cy) ** 2 > 16 for k in keep):
            keep.append(d)
    return [{"x": round(float(d.cx), 1), "y": round(float(d.cy), 1),
             "cls": None, "pred": getattr(d, "class_name", None)} for d in keep]


# Ring colours for the canvas, keyed by the name `legend` already assigns each
# row. `LegendEntry.marker` is the cropped glyph image, not a colour, so reading a
# pixel out of it returns the panel background rather than the ink.
_RING = {"red": "#e5352b", "orange": "#f08a24", "yellow": "#e8c020",
         "green": "#3aa757", "cyan": "#25c4d6", "blue": "#2b6fe0",
         "magenta": "#d436b8", "grey": "#9aa0a6", "gray": "#9aa0a6"}


def _glyph_png(dialog_rgb, e, pad=11) -> str:
    """The row's actual marker, cropped from the dialog and blown up.

    The class palette shows this rather than a colour swatch because colour alone
    cannot tell two classes apart — the hard cases in this dataset share a colour
    and differ only in shape. The labeller has to compare the glyph on the aerial
    against the glyph in the legend, so the legend's real pixels are what belongs
    in the palette.
    """
    try:
        cy, cx = int(e.cy), int(e.cx)
        h, w = dialog_rgb.shape[:2]
        crop = dialog_rgb[max(0, cy - pad):min(h, cy + pad + 1),
                          max(0, cx - pad):min(w, cx + pad + 1)]
        if crop.size == 0:
            return ""
        big = cv2.resize(crop, (66, 66), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        return "data:image/png;base64," + base64.b64encode(buf).decode() if ok else ""
    except Exception:                                              # noqa: BLE001
        return ""


def legend_for(shot):
    """Class options, glyph images and swatch colours from the image's own dialog.

    Per-image by construction: (shape, colour) -> class is not global in this
    dataset — "Site" is a filled circle on one image and an asterisk on the next —
    so the options offered must come from this screenshot's legend and never from
    a fixed vocabulary.
    """
    try:
        entries, bbox = parse_screenshot(shot)
        dialog = None
        if bbox is not None:
            x, y, w, h = bbox
            dialog = shot[y:y + h, x:x + w]
            try:
                attach_class_names(dialog, entries)
            except Exception:                                      # noqa: BLE001
                pass
        names, colors, glyphs = [], {}, {}
        for i, e in enumerate(entries):
            name = (e.class_name or "").strip() or f"row {i + 1} ({e.color})"
            if name in colors:
                name = f"{name} #{i + 1}"
            names.append(name)
            colors[name] = _RING.get((e.color or "").lower(), "#9aa0a6")
            glyphs[name] = _glyph_png(dialog, e) if dialog is not None else ""

        # The whole dialog, so the labeller can check a class name against the
        # source. Count-OCR reads at 60-65% and names are fuzzy-matched, so the
        # parsed list is not authoritative — the picture is.
        dlg = ""
        if dialog is not None and dialog.size:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(dialog, cv2.COLOR_RGB2BGR))
            if ok:
                dlg = "data:image/png;base64," + base64.b64encode(buf).decode()
        return names, colors, glyphs, bbox, entries, dlg
    except Exception:                                              # noqa: BLE001
        return [], {}, {}, None, [], ""


def select(bench: pd.DataFrame, n: int, only: str | None) -> pd.DataFrame:
    bench = bench.assign(name=bench.screenshot_key.map(os.path.basename))
    if only:
        return bench[bench["name"] == only]
    pinned = bench[bench["name"].isin(PINNED)]
    rest = bench[~bench["name"].isin(PINNED)]
    # Fill the remainder by taking one frame from each band in turn, so a request
    # for n frames yields n and no single density dominates the remainder.
    queues = [list(g.itertuples()) for _, g in rest.groupby("band")]
    picked, i = [], 0
    while len(picked) < max(0, n - len(pinned)) and any(queues):
        q = queues[i % len(queues)]
        if q:
            picked.append(q.pop(0))
        elif all(not x for x in queues):
            break
        i += 1
    extra = pd.DataFrame(picked).drop(columns=["Index"], errors="ignore") if picked \
        else rest.head(0)
    return pd.concat([pinned, extra]).head(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--blind", type=int, default=2,
                    help="frames left unseeded, as a control on seeding bias")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs("data/labels", exist_ok=True)
    template = open(TEMPLATE, encoding="utf-8").read()
    cache = json.load(open(ALIGN_CACHE)) if os.path.exists(ALIGN_CACHE) else {}
    chosen = select(pd.read_csv(BENCHMARK), args.frames, args.only)

    # Blind controls go on the *smallest* frames available. Labelling unseeded is
    # slow, and putting the control on a 673-dot colony would cost hours to answer
    # a question a 40-dot frame answers just as well: whether a labeller shown the
    # detector's output finds fewer markers than one working from scratch.
    pool = chosen[~chosen["name"].isin(PINNED)] if len(chosen) > args.blind else chosen
    blind_set = set(pool.nsmallest(args.blind, "dots")["name"]) if args.blind else set()

    for i, r in enumerate(chosen.itertuples()):
        name = os.path.basename(r.screenshot_key)
        shot, orig = read(r.screenshot_key), read(r.highres_key)
        if shot is None:
            print(f"  {name}: screenshot missing, skipped")
            continue

        blind = name in blind_set
        classes, colors, glyphs, bbox, entries, dialog_png = legend_for(shot)
        # Always available, so a frame whose legend fails to parse (00097 yields
        # zero rows) can still be labelled for position, and so a marker the
        # legend does not explain has somewhere to go instead of being forced
        # into a wrong class.
        classes = classes + ["unclear / not in legend"]
        colors["unclear / not in legend"] = "#ffffff"
        if bbox is None:
            try:
                bbox = locate_dialog(shot)
            except Exception:                                      # noqa: BLE001
                bbox = None
        res = cached_align(r.screenshot_key, shot, orig, cache) if orig is not None else None
        seeds = [] if blind else seeds_for(shot, orig, res, bbox, entries)

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(shot, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            print(f"  {name}: could not encode, skipped")
            continue

        # Both keys travel with the labels. Deriving the original's path later by
        # string-replacing "screenshots" would break on any colony whose name
        # happens to contain it.
        data = dict(frame=name, key=r.screenshot_key, highres_key=r.highres_key,
                    truth=float(r.dots), band=r.band,
                    classes=classes, colors=colors, glyphs=glyphs, seeds=seeds,
                    dialog=dialog_png, dialog_bbox=list(bbox) if bbox is not None else None,
                    image="data:image/jpeg;base64," + base64.b64encode(buf).decode())
        html = (template.replace("__DATA__", json.dumps(data))
                        .replace("__FRAME__", name)
                        .replace("__TRUTH__", str(int(r.dots))))
        path = os.path.join(OUT, name.replace(".jpg", "") + ".html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  {name:34s} {r.band:7s} truth={r.dots:6.0f}  "
              f"seeds={len(seeds):5d}  classes={len(classes):2d}"
              f"{'   [BLIND control]' if blind else ''}")

    print(f"\nwrote {len(chosen)} pages to {OUT}/")
    print("Open one, click through it, press S, and save the .json into data/labels/.")


if __name__ == "__main__":
    main()
