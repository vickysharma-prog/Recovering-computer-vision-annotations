"""
Score detection and classification against hand-labelled dots.

This is the measurement `eval_detection.py` cannot make. That script compares a
count to a count, and a count hides everything that matters: "61 detected against
61 true" is the same number whether all 61 markers were found, or 40 were found
and 21 invented, or 61 were placed on empty water. Here every detection is matched
to a labelled marker, so misses, false positives and placement error separate out.

Classification is scored on the same matched pairs. The existing 0.36 agreement
figure is per-class *count* agreement, which cannot see a label swap — the known
failure is two same-colour classes trading labels wholesale while their counts
stay plausible (WHIB site against WHIB bird, 232/86 true, 89/254 produced). A
per-dot comparison against a human label sees exactly that, and the confusion
table below names the pairs responsible.

Usage:
    python scripts/eval_localisation.py
    python scripts/eval_localisation.py --tol 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import align, AlignResult
from src.classify import detect_dots, detect_dots_subtract, assign_classes
from src.legend import locate_dialog, parse_screenshot, attach_class_names
from src.select import DEFAULT_QUALITY, accept_frame
# Shared with the labelling page so both derive a row's display name the same way;
# deriving it twice is how a stored `cls` string stops resolving to its row.
from scripts.label_dots import legend_options

LABELS = "data/labels"
PAIR_DIR = "data/fixtures/pairs"
ALIGN_CACHE = "data/cache/align_cache.json"


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


def match(pred_xy, true_xy, tol):
    """Greedy one-to-one pairing of detections to labels, closest first.

    Greedy on a global distance ordering rather than per-detection nearest: the
    latter lets two detections claim the same marker, which would report a miss
    and a false positive that are really one duplicate.
    """
    pairs = sorted(((np.hypot(px - tx, py - ty), i, j)
                    for i, (px, py) in enumerate(pred_xy)
                    for j, (tx, ty) in enumerate(true_xy)
                    if abs(px - tx) <= tol and abs(py - ty) <= tol),
                   key=lambda p: p[0])
    used_p, used_t, out = set(), set(), []
    for d, i, j in pairs:
        if d > tol or i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); out.append((i, j, d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=5.0,
                    help="px within which a detection counts as finding a marker")
    ap.add_argument("--quality", type=float, default=DEFAULT_QUALITY,
                    help="frame-selection quality target (src/select.py). Only "
                         "selected frames reach classification and the exported "
                         "dataset, so the selected subtotal is what the pipeline "
                         "actually produces.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(LABELS, "*.json")))
    if not files:
        print(f"No label files in {LABELS}/.")
        print("Run: python scripts/label_dots.py   then label and save the JSON there.")
        return

    cache = json.load(open(ALIGN_CACHE)) if os.path.exists(ALIGN_CACHE) else {}
    rows, confusion = [], Counter()

    for path in files:
        lab = json.load(open(path))
        shot = read(lab["screenshot_key"])
        orig = read(lab.get("highres_key") or
                    lab["screenshot_key"].replace("screenshots", "high_resolution_photos"))
        if shot is None:
            print(f"  {lab['frame']}: screenshot missing, skipped")
            continue

        true_xy = [(d["x"], d["y"]) for d in lab["dots"]]
        true_cls = [d.get("cls") for d in lab["dots"]]

        try:
            bbox = locate_dialog(shot)
        except Exception:                                          # noqa: BLE001
            bbox = None
        res = cached_align(lab["screenshot_key"], shot, orig, cache) if orig is not None else None
        if res is not None and res.ok:
            dots, path_used = detect_dots_subtract(shot, orig, res, exclude=bbox), "subtract"
        else:
            dots, path_used = detect_dots(shot, exclude=bbox), "colour"

        pred_xy = [(d.cx, d.cy) for d in dots]
        m = match(pred_xy, true_xy, args.tol)
        tp = len(m)
        prec = tp / len(pred_xy) if pred_xy else 0.0
        rec = tp / len(true_xy) if true_xy else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        err = float(np.median([d for _i, _j, d in m])) if m else float("nan")

        # Would frame selection keep this frame? Only selected frames reach
        # classification and the exported dataset, so the selected subtotal is what
        # the pipeline actually produces. See src/select.py for the bound.
        survey = lab.get("survey_count") or len(true_xy)
        decision = accept_frame(len(pred_xy), survey, quality=args.quality)
        ratio, accepted = decision.ratio, decision.accepted

        # Classification on the matched pairs only: a class score on a dot that
        # was never found would be measuring detection, not classification.
        cls_n = cls_ok = cls_unscoreable = name_ok = 0
        true_row = [d.get("row") for d in lab["dots"]]
        if any(true_cls):
            try:
                entries, dbox = parse_screenshot(shot)
                if dbox is not None:
                    x, y, w, h = dbox
                    try:
                        attach_class_names(shot[y:y + h, x:x + w], entries)
                    except Exception:                              # noqa: BLE001
                        pass
                # Sets `class_name` and `legend_row` on each dot in place; `dots`
                # stays in the same order as `pred_xy`, so match indices line up.
                assign_classes(dots, entries)

                # The legend ROW is the class identity, not the name. Names come
                # from OCR: they are absent on 26% of rows corpus-wide, and two
                # rows on one dialog can carry the same string ('ad' twice on
                # 0115). Scoring by name therefore both misses correct answers
                # (truth frozen as "row 2 (green)" while the prediction is None)
                # and credits wrong ones (a dot on the wrong of two 'ad' rows).
                # Older label files store only `cls`, so map the stored string
                # back through the labelling page's own naming.
                # Local names: `rows` is the outer per-frame accumulator.
                opt_names, opt_rows = legend_options(entries)
                by_name = dict(zip(opt_names, opt_rows))

                for i, j, _d in m:
                    want_name = (true_cls[j] or "").strip()
                    if not want_name:
                        continue
                    want_row = true_row[j]
                    if want_row is None:
                        want_row = by_name.get(want_name)
                    have_row = getattr(dots[i], "legend_row", None) if i < len(dots) else None
                    have_name = (getattr(dots[i], "class_name", None) or "").strip() \
                        if i < len(dots) else ""
                    if want_row is None:
                        # "unclear / not in legend", or a name the current parse no
                        # longer produces. Not a matcher failure; counted apart so
                        # it cannot silently deflate the accuracy.
                        cls_unscoreable += 1
                        continue
                    cls_n += 1
                    if have_row == want_row:
                        cls_ok += 1
                    else:
                        confusion[(want_name, have_name or "none")] += 1
                    if have_name == want_name:
                        name_ok += 1
            except Exception as exc:                               # noqa: BLE001
                print(f"  {lab['frame']}: classification skipped ({exc})")

        swept = lab.get("tiles_reviewed", 0), lab.get("tiles_total", 0)
        rows.append(dict(frame=lab["frame"], band=lab.get("band", "?"), path=path_used,
                         labels=len(true_xy), detected=len(pred_xy), tp=tp,
                         fp=len(pred_xy) - tp, fn=len(true_xy) - tp,
                         precision=prec, recall=rec, f1=f1, loc_err=err,
                         survey=lab.get("survey_count"), seeded=lab.get("seeded"),
                         cls_scored=cls_n, cls_correct=cls_ok,
                         cls_name_correct=name_ok, cls_unscoreable=cls_unscoreable,
                         ratio=ratio, accepted=accepted,
                         swept=f"{swept[0]}/{swept[1]}"))
        cls_txt = f"cls={cls_ok}/{cls_n}" if cls_n else "cls=  -  "
        print(f"  {lab['frame'][:32]:34s} {path_used:8s} labels={len(true_xy):5d} "
              f"det={len(pred_xy):5d}  P={prec:.2f} R={rec:.2f} F1={f1:.2f}  "
              f"err={err:4.1f}px  {cls_txt:12s} swept {swept[0]}/{swept[1]}")

    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    print(f"POOLED over {len(df)} frames, tolerance {args.tol:.0f}px")
    print(f"  precision {P:.3f}   recall {R:.3f}   "
          f"F1 {2*P*R/(P+R) if P+R else 0:.3f}")
    print(f"  {tp} found, {fp} false positives, {fn} missed")
    print(f"  median placement error {df.loc_err.median():.2f}px")

    if df.cls_scored.sum():
        n_cls = int(df.cls_scored.sum())
        acc = df.cls_correct.sum() / n_cls
        name_acc = df.cls_name_correct.sum() / n_cls
        print(f"\nCLASSIFICATION on {n_cls} matched, class-labelled dots")
        # Row accuracy is the real figure: did the dot land on the right legend
        # row. Name accuracy is the same dots scored the old way, by comparing
        # OCR'd strings, and is reported alongside so the change in the headline
        # is visibly a change of metric and not a change in the pipeline.
        print(f"  per-dot accuracy, legend ROW  {acc:.3f}   ({int(df.cls_correct.sum())}/{n_cls})")
        print(f"  same dots scored by NAME      {name_acc:.3f}   "
              f"({int(df.cls_name_correct.sum())}/{n_cls})")
        if df.cls_unscoreable.sum():
            print(f"  excluded as unscoreable: {int(df.cls_unscoreable.sum())} "
                  f"(labelled 'unclear / not in legend', or a name this parse no "
                  f"longer produces)")
        print("\n  per frame:")
        for _, r in df.iterrows():
            if not r.cls_scored:
                continue
            print(f"    {r.frame[:32]:34s} row {r.cls_correct:4d}/{r.cls_scored:<4d} "
                  f"= {r.cls_correct/r.cls_scored:.3f}    name "
                  f"{r.cls_name_correct/r.cls_scored:.3f}")
        print("\n  worst confusions (true -> predicted):")
        for (want, have), n in confusion.most_common(8):
            print(f"    {n:5d}  {want:28s} -> {have}")

    # Seeded frames can only flatter recall, since a labeller shown the detector's
    # output confirms it more readily than they hunt for what it missed. The blind
    # frames are the control; if they carry materially more labels per frame, the
    # seeded recall above is optimistic.
    if df.seeded.nunique() > 1:
        print("\nSEEDING CONTROL (labels per frame)")
        for seeded, g in df.groupby("seeded"):
            print(f"  {'seeded' if seeded else 'blind ':7s} n={len(g)}  "
                  f"median labels {g.labels.median():.0f}  median recall {g.recall.median():.2f}")

    print("\nby band:")
    for band, g in df.groupby("band"):
        print(f"  {band:8s} n={len(g)}  P={g.precision.median():.2f}  "
              f"R={g.recall.median():.2f}  F1={g.f1.median():.2f}")

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/eval_localisation.csv", index=False)
    print("\nwrote results/eval_localisation.csv")


if __name__ == "__main__":
    main()
