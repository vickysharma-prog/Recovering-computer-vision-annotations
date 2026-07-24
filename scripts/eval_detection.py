"""
Phase 2 gate: does difference-based detection beat colour-based detection?

Usage:
    python scripts/eval_detection.py [--limit N]

Compares, per benchmark image, the number of markers found by:
  * OLD — `classify.detect_dots`, colour thresholds on the screenshot alone.
  * NEW — subtraction against the aligned clean original.

against `category_sum`, the sum of the per-dot-type survey columns. That is the
ground truth because it matches the counting tool's own "Total Count" field;
`total_birds` does NOT (it excludes chicks, and undercounts dots by up to 57% on a
single image). Every earlier over-detection figure we quoted was measured against
`total_birds` and is therefore unreliable.

The counts here are deliberately free-standing: no per-class top-N selection is
applied, because selecting toward a known count makes the count match by
construction and proves nothing.

Ratio, not difference, is the headline: over-detection is multiplicative.
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
from src.subtract import extract_annotations, dot_candidates
from src.classify import detect_dots
from src.legend import locate_dialog

BENCHMARK = "data/cache/benchmark.csv"
PAIR_DIR = "data/fixtures/pairs"
ALIGN_CACHE = "data/cache/align_cache.json"


def read(key: str):
    path = os.path.join(PAIR_DIR, key.replace("avian_monitoring/", ""))
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_cache() -> dict:
    if os.path.exists(ALIGN_CACHE):
        with open(ALIGN_CACHE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(ALIGN_CACHE), exist_ok=True)
    with open(ALIGN_CACHE, "w") as f:
        json.dump(cache, f)


def cached_align(key: str, shot, orig, cache: dict) -> AlignResult:
    """Align, reusing a stored transform when one exists.

    SIFT over a ~4752px original dominates the runtime, and the registration does
    not change while subtraction parameters are being tuned — so caching it turns
    an eight-minute sweep into seconds. Delete data/cache/align_cache.json after
    touching src/align.py.
    """
    hit = cache.get(key)
    if hit is not None:
        return AlignResult(
            H=np.array(hit["H"], np.float64) if hit["H"] is not None else None,
            scale=hit["scale"], matches=hit["matches"], inliers=hit["inliers"],
            reproj_err=hit["reproj_err"], model=hit["model"], ok=hit["ok"],
            reason=hit["reason"])
    res = align(shot, orig)
    cache[key] = dict(H=res.H.tolist() if res.H is not None else None,
                      scale=res.scale, matches=res.matches, inliers=res.inliers,
                      reproj_err=res.reproj_err, model=res.model, ok=res.ok,
                      reason=res.reason)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute alignment (required after editing src/align.py)")
    args = ap.parse_args()

    bench = pd.read_csv(BENCHMARK)
    if args.limit:
        bench = bench.head(args.limit)

    cache = {} if args.no_cache else _load_cache()
    rows = []
    for i, r in enumerate(bench.itertuples(), 1):
        shot, orig = read(r.screenshot_key), read(r.highres_key)
        if shot is None or orig is None:
            continue

        # OLD: colour detection, excluding the dialog when it can be located.
        try:
            bbox = locate_dialog(shot)
            old = len(detect_dots(shot, exclude=bbox))
        except Exception:                                          # noqa: BLE001
            old = None

        # NEW: subtraction. Falls back to the colour path when alignment fails,
        # which is the behaviour we would ship.
        res = cached_align(r.screenshot_key, shot, orig, cache)
        if res.ok:
            new = len(dot_candidates(extract_annotations(shot, orig, res)))
            path = "subtract"
        else:
            new, path = old, "colour-fallback"

        rows.append(dict(name=os.path.basename(r.screenshot_key), year=r.year,
                         band=r.band, truth=r.dots, old=old, new=new, path=path))
        print(f"  [{i:>2}] {os.path.basename(r.screenshot_key)[:30]:32}"
              f"truth={r.dots:>5}  old={old if old is not None else '-':>6}"
              f"  new={new:>6}  {path}")

    if not args.no_cache:
        _save_cache(cache)

    df = pd.DataFrame(rows)
    df = df[df.truth > 0]
    df["old_ratio"] = df.old / df.truth
    df["new_ratio"] = df.new / df.truth

    print("\n" + "=" * 76)
    print(f"images {len(df)}   (subtraction used on {(df.path=='subtract').sum()}, "
          f"colour fallback on {(df.path=='colour-fallback').sum()})")
    print("\ndetected / truth  — 1.0 is perfect, >1 over-detects, <1 misses")
    print(f"  OLD colour   median {df.old_ratio.median():6.2f}x   "
          f"mean {df.old_ratio.mean():6.2f}x")
    print(f"  NEW subtract median {df.new_ratio.median():6.2f}x   "
          f"mean {df.new_ratio.mean():6.2f}x")
    print(f"\nmedian |log2 ratio| (symmetric, lower better): "
          f"old {np.abs(np.log2(df.old_ratio.replace(0, np.nan))).median():.2f}  "
          f"new {np.abs(np.log2(df.new_ratio.replace(0, np.nan))).median():.2f}")

    # The |log2| median silently drops zero-detection frames (log2(0)=nan), so a
    # lever that pushes a struggling frame to 0 can *improve* the median while
    # making detection worse. Track misses and zeros alongside it so that trade
    # is visible rather than hidden.
    def _health(col: str) -> str:
        r = df[col]
        return (f"miss(<0.5x) {int((r < 0.5).sum()):>2}   "
                f"zero {int((r == 0).sum()):>2}   over(>2x) {int((r > 2).sum()):>2}")
    print(f"  OLD: {_health('old_ratio')}")
    print(f"  NEW: {_health('new_ratio')}")

    for key in ("band", "year"):
        print(f"\nby {key}:   n   old_ratio   new_ratio")
        for k, g in df.groupby(key):
            print(f"  {str(k)[:10]:12}{len(g):>4}{g.old_ratio.median():>11.2f}x"
                  f"{g.new_ratio.median():>11.2f}x")

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/eval_detection.csv", index=False)
    print("\nwrote results/eval_detection.csv")


if __name__ == "__main__":
    main()
