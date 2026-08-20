"""
Export the recovered annotations as a DeepForest training dataset.

This is the stage that turns a measured pipeline into something another person can
use. Everything upstream works in the screenshot's pixel grid; a model trains on the
clean originals. `src/mapping.py` moves the coordinates, and this script decides which
frames are in, what each row's label says, and how much of the pipeline's own
uncertainty travels with it.

## Which frames

The set is recomputed here rather than listed, because "the 25 frames" has been quoted
for weeks without living in any code:

    60  cached pairs                       (63 in the benchmark, 60 downloaded)
    31  pass frame selection               src/select.accept_frame
    28  a dialog is found                  src/legend.locate_dialog
    25  the dialog box is correct          minus three named frames

`results/dataset/frames.csv` records every cached frame and why it is in or out, so
the funnel can be checked without rerunning anything.

## What a row says

The label policy is "export every dot, and say which labels are real":

    species resolved      label = "LAGU site"   species_resolved = true
    row but no species    label = "Bird"        species_resolved = false

Dropping the second group would throw away good coordinates over a text problem that
detection does not care about — DeepForest's bird model trains on a single class.
Writing "Bird" without saying so would hide which species are real. `frame` and
`legend_row` travel on every row, so a reviewer who fixes one legend row fixes every
dot on it: 22 rows rather than 1,500 dots.

A dot that matched **no** row is a different case. That is the pipeline's own
valid/invalid decision — it is saying the detection is not a marker — so it is kept in
`annotations_full.csv` with `assigned=false` and left out of the DeepForest file.

## Boxes are provisional

The survey recorded points, not extents, so no box size is recoverable from the data.
`--box` sets one, default 100 original pixels, and every row carries
`box_provisional=true`. The default is not arbitrary: the earlier prototype's model
predicted boxes near 106x105, and `learnings.md` #16 records that 80x80 ground truth
against that gives IoU 0.45-0.55, low enough that half the correct predictions earn no
credit during training. Which size is right is settled by the training sweep, not here.

Usage:
    python scripts/export_dataset.py
    python scripts/export_dataset.py --box 80 --out results/dataset_box80
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import align, AlignResult
from src.birdsize import box_from_size, frame_bird_size
from src.classify import detect_dots, detect_dots_subtract, assign_classes
from src.legend import locate_dialog, parse_screenshot, attach_class_names
from src.mapping import map_dots
from src.select import DEFAULT_QUALITY, accept_frame

BENCHMARK = "data/cache/benchmark.csv"
PAIR_DIR = "data/fixtures/pairs"
ALIGN_CACHE = "data/cache/align_cache.json"

# Excluded by name, not by threshold. `locate_dialog` returns a box on these three
# that is wrong: 0507 and 3824 sit over the aerial, 0465 runs too tall. Six automatic
# tests were measured to separate them from the good boxes and every one overlapped,
# so naming three frames is more honest than a threshold fitted to three examples.
# Full names read off the benchmark, not written from memory: the docs record only the
# trailing number, and a guessed prefix matches nothing and silently excludes nothing.
# Only 0507 actually reaches this test. 3824 (1.72x) and 0465 (2.85x) are already gone
# at frame selection, so the documented "28 with a dialog, minus three, leaves 25"
# subtracts two frames twice.
BAD_DIALOG = ("14June10Camera1-Card1-0507", "20May18Camera2-Card1-3824",
              "23May13Camera1-Card1-0465")


def read(key: str):
    path = os.path.join(PAIR_DIR, key.replace("avian_monitoring/", ""))
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cached_align(key, shot, orig, cache):
    """Reuse the measured transform; `align` is slow and already scored at 0.38px."""
    hit = cache.get(key)
    if hit is None:
        return align(shot, orig)
    return AlignResult(
        H=np.array(hit["H"], np.float64) if hit["H"] is not None else None,
        scale=hit["scale"], matches=hit["matches"], inliers=hit["inliers"],
        reproj_err=hit["reproj_err"], model=hit["model"], ok=hit["ok"],
        reason=hit["reason"])


def frame_name(key: str) -> str:
    return os.path.splitext(os.path.basename(key))[0]


def dialog_box(shot):
    """Where the dialog sits, or None. Cheap, and needed on every frame.

    Detection has to exclude it — the legend's own markers are dots too, and counting
    them would corrupt both the detected total and the frame-selection ratio built
    from it.
    """
    try:
        return locate_dialog(shot)
    except Exception:                                              # noqa: BLE001
        return None


def legend_for(shot):
    """Parse the dialog's rows and read their text. Returns the entries, possibly [].

    Only called for frames that survive the funnel. `attach_class_names` runs
    Tesseract twice per row, which is by far the slowest thing here, and a frame that
    was already rejected has nothing to spend it on.
    """
    try:
        entries, bbox = parse_screenshot(shot)
    except Exception:                                              # noqa: BLE001
        return []
    if bbox is not None and entries:
        x, y, w, h = bbox
        try:
            attach_class_names(shot[y:y + h, x:x + w], entries)
        except Exception:                                          # noqa: BLE001
            pass          # names are cosmetic here; the row is the identity
    return entries


def box_for(x: float, y: float, box: int, w: int, h: int):
    """A square box centred on the dot, clipped to the image."""
    half = box / 2.0
    xmin, ymin = max(0.0, x - half), max(0.0, y - half)
    xmax, ymax = min(float(w - 1), x + half), min(float(h - 1), y + half)
    return xmin, ymin, xmax, ymax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=0,
                    help="force one box side in ORIGINAL pixels for every frame. "
                         "The default, 0, measures each frame's own birds instead "
                         "(src/birdsize.py). Use a fixed value only to produce the "
                         "comparison arms of the training sweep.")
    ap.add_argument("--quality", type=float, default=DEFAULT_QUALITY,
                    help="frame-selection quality target (src/select.py)")
    ap.add_argument("--out", default="results/dataset")
    args = ap.parse_args()

    if not os.path.exists(BENCHMARK):
        print(f"No benchmark at {BENCHMARK}. Run scripts/build_benchmark.py first.")
        return
    os.makedirs(args.out, exist_ok=True)

    bench = pd.read_csv(BENCHMARK)
    cache = json.load(open(ALIGN_CACHE)) if os.path.exists(ALIGN_CACHE) else {}

    frames, rows = [], []

    def finish(record):
        """Record a frame and print its one line.

        Every frame prints, not only the exported ones. Two thirds of the benchmark
        is dropped by design, and a run that stays silent through them looks hung —
        it is also the funnel itself, so it is worth seeing go past.
        """
        frames.append(record)
        state = record["excluded_reason"] or "exported"
        print(f"  {record['frame'][:34]:36s} det={record['detected']:5d} "
              f"rows={record['legend_rows']:3d} out={record['dots_exported']:5d}"
              f"  {state}", flush=True)

    for _i, b in bench.iterrows():
        key, name = b["screenshot_key"], frame_name(b["screenshot_key"])
        record = dict(frame=name, band=b.get("band"), survey_count=b.get("dots"),
                      cached=False, detected=0, ratio=None, accepted=False,
                      dialog_found=False, legend_rows=0, dots_assigned=0,
                      dots_exported=0, dropped_out_of_bounds=0,
                      legend_count_sum=None, legend_rows_counted=0,
                      bird_px=None, box_px=None, box_measured=False,
                      excluded_reason="")

        shot = read(key)
        orig = read(b["highres_key"])
        if shot is None or orig is None:
            record["excluded_reason"] = "not cached locally"
            finish(record)
            continue
        record["cached"] = True

        # Locating the dialog comes first because detection must exclude it. Reading
        # it is deferred until the frame has survived the funnel.
        bbox = dialog_box(shot)
        record["dialog_found"] = bbox is not None

        res = cached_align(key, shot, orig, cache)
        if res.ok:
            dots, path_used = detect_dots_subtract(shot, orig, res, exclude=bbox), "subtract"
        else:
            dots, path_used = detect_dots(shot, exclude=bbox), "colour"
        record["detected"] = len(dots)
        record["path"] = path_used

        # Frame selection, on the ratio alone. Needs no labels: a true positive cannot
        # outnumber the dots present or the dots detected, so one quality target fixes
        # a two-sided band. See src/select.py.
        survey = b.get("dots")
        decision = accept_frame(len(dots), survey, quality=args.quality)
        record["ratio"], record["accepted"] = decision.ratio, decision.accepted
        if not decision.accepted:
            record["excluded_reason"] = f"frame selection: {decision.reason}"
            finish(record)
            continue
        if bbox is None:
            record["excluded_reason"] = "no dialog found"
            finish(record)
            continue
        if name in BAD_DIALOG:
            record["excluded_reason"] = "dialog box wrong (excluded by name)"
            finish(record)
            continue
        if not res.ok:
            # Without a transform there is nowhere to map the dots to. Colour-path
            # detection is still fine for scoring, but it cannot be exported.
            record["excluded_reason"] = f"alignment rejected: {res.reason}"
            finish(record)
            continue

        # Survived the funnel, so it is worth reading the dialog properly now.
        entries = legend_for(shot)
        record["legend_rows"] = len(entries)
        # What the dialog itself says, read off the image. Distinct from
        # survey_count, which comes from the published CSV and which the pipeline
        # never reads: comparing the two says whether the count OCR is working
        # without going outside the image for the answer.
        counted = [e.count for e in entries if e.count is not None]
        record["legend_count_sum"] = sum(counted) if counted else None
        record["legend_rows_counted"] = len(counted)
        if not entries:
            record["excluded_reason"] = "legend did not parse"
            finish(record)
            continue

        assign_classes(dots, entries)
        mapped = map_dots(dots, res, orig.shape)
        h, w = orig.shape[:2]
        rel_path = b["highres_key"].replace("avian_monitoring/", "")

        # Box size is measured from this frame's own birds unless one was forced on
        # the command line. The same species is 11px across on one photograph and
        # 21px on another, because eleven years of surveys flew different cameras at
        # different heights, so one number cannot serve them all. See src/birdsize.py.
        if args.box:
            bird_px, box, measured = None, args.box, False
        else:
            est = frame_bird_size(orig, [(m.x, m.y) for m in mapped if m.in_bounds])
            bird_px = round(est.median_px, 2) if est.ok else None
            box = box_from_size(est.median_px) if est.ok else None
            measured = est.ok
        record["bird_px"], record["box_px"], record["box_measured"] = \
            bird_px, box, measured

        for m in mapped:
            d = dots[m.index]
            assigned = d.legend_row is not None
            record["dots_assigned"] += int(assigned)
            if not m.in_bounds:
                record["dropped_out_of_bounds"] += 1

            exported = bool(m.in_bounds and assigned)
            record["dots_exported"] += int(exported)
            species_resolved = d.species is not None
            rows.append(dict(
                image_path=rel_path,
                # Filled once every frame has been measured, so a frame whose own
                # birds could not be measured can borrow the corpus median rather
                # than being dropped over it.
                xmin=None, ymin=None, xmax=None, ymax=None,
                bird_px=bird_px, box_px=box, box_measured=measured,
                img_w=w, img_h=h,
                label=(d.class_name if species_resolved and d.class_name else "Bird"),
                species_resolved=species_resolved,
                frame=name, legend_row=d.legend_row,
                class_name=d.class_name, species=d.species, category=d.category,
                match_score=d.match_score,
                candidates="|".join(str(c) for c in d.candidates),
                candidate_scores="|".join(f"{s:g}" for s in d.candidate_scores),
                x_orig=round(m.x, 2), y_orig=round(m.y, 2),
                shot_x=round(m.shot_x, 2), shot_y=round(m.shot_y, 2),
                in_bounds=m.in_bounds, assigned=assigned, exported=exported,
                box_provisional=True,
                align_model=res.model, align_reproj_err=round(res.reproj_err, 3),
                band=b.get("band"), year=b.get("year"), colony=b.get("colony"),
            ))

        finish(record)

    if not rows:
        print("Nothing exported.")
        return

    full = pd.DataFrame(rows)
    frames_df = pd.DataFrame(frames)

    # A frame whose own birds could not be measured borrows the corpus median rather
    # than being dropped. Two of the 25 fail, both sparse enough that too few dots
    # sit far enough from the image edge to measure: `220` holds 8 dots and `00825`
    # holds 7. Borrowing is marked with box_measured=False on every row.
    if not args.box:
        fallback = box_from_size(float(full.loc[full.box_px.notna(), "bird_px"]
                                       .median()))
        full["box_px"] = full["box_px"].fillna(fallback)
        frames_df["box_px"] = frames_df["box_px"].fillna(fallback)
        print(f"\nfallback box for unmeasurable frames: {fallback}px")

    # Boxes are laid out only now, because the size was not known during the pass.
    half = full["box_px"] / 2.0
    full["xmin"] = (full.x_orig - half).clip(lower=0).round(2)
    full["ymin"] = (full.y_orig - half).clip(lower=0).round(2)
    full["xmax"] = np.minimum(full.x_orig + half, full.img_w - 1).round(2)
    full["ymax"] = np.minimum(full.y_orig + half, full.img_h - 1).round(2)

    exported = full[full["exported"]]

    full_path = os.path.join(args.out, "annotations_full.csv")
    df_path = os.path.join(args.out, "annotations_deepforest.csv")
    frames_path = os.path.join(args.out, "frames.csv")
    full.to_csv(full_path, index=False)
    # DeepForest reads a fixed six-column schema; extra columns are a hazard in
    # someone else's loader. Both files come from one pass so they cannot drift.
    exported[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]] \
        .to_csv(df_path, index=False)
    frames_df.to_csv(frames_path, index=False)

    # frames.csv carries every candidate with the reason it is in or out, which is
    # what makes the funnel auditable but means most of its rows are frames that did
    # not ship. This is the shipped set on its own, so nobody has to filter for it.
    exported_path = os.path.join(args.out, "exported_frames.csv")
    frames_df[frames_df.dots_exported > 0].to_csv(exported_path, index=False)

    cached = frames_df["cached"].sum()
    accepted = frames_df["accepted"].sum()
    with_dialog = (frames_df["accepted"] & frames_df["dialog_found"]).sum()
    used = frames_df["dots_exported"].gt(0).sum()
    n_res = int(exported["species_resolved"].sum())

    print("\nFunnel")
    print(f"  benchmark frames        {len(frames_df)}")
    print(f"  cached locally          {cached}")
    print(f"  pass frame selection    {accepted}")
    print(f"  a dialog is found       {with_dialog}")
    print(f"  exported                {used}")
    print("\nDots")
    print(f"  detected on those frames {int(full.shape[0])}")
    print(f"  assigned to a legend row {int(full['assigned'].sum())}")
    print(f"  exported                 {len(exported)}")
    print(f"  dropped, out of bounds   {int(frames_df['dropped_out_of_bounds'].sum())}")
    print(f"\n  species resolved  {n_res}/{len(exported)} = "
          f"{n_res / len(exported):.3f}   the rest are labelled 'Bird'")

    # Box sizes, over the whole set rather than any one frame. Both averages are
    # printed because they answer different questions: per frame is what the method
    # produced, per dot is what the model will actually see, and the dense frames
    # carry most of the dots.
    ex_frames = frames_df[frames_df.dots_exported > 0]
    if args.box:
        print(f"\nBox               forced to {args.box}px on every frame")
    else:
        bp = ex_frames["box_px"]
        print(f"\nBox, measured per frame from that frame's own birds")
        print(f"  bird size    median {ex_frames['bird_px'].median():.1f}px   "
              f"range {ex_frames['bird_px'].min():.1f}-{ex_frames['bird_px'].max():.1f}px")
        print(f"  box per frame   mean {bp.mean():.1f}px   median {bp.median():.0f}px   "
              f"range {bp.min():.0f}-{bp.max():.0f}px")
        print(f"  box per dot     mean {exported['box_px'].mean():.1f}px   "
              f"median {exported['box_px'].median():.0f}px")
        print(f"  measured on {int(ex_frames['box_measured'].sum())}/{len(ex_frames)} "
              f"frames; the rest borrow the corpus median")
    print(f"\n  {df_path}\n  {full_path}\n  {frames_path}")
    print(f"  DeepForest root_dir is {PAIR_DIR}")


if __name__ == "__main__":
    main()
