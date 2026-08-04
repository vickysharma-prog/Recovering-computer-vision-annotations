"""
Phase 0 gate, part 2: does the survey's `total_birds` equal the dot count the
counting tool itself recorded?

Usage:
    python scripts/probe_totalcount.py [--n 8]

The "Manual Point Count" dialog baked into every screenshot displays a
`Total Count` field — the tool's own tally of dots placed on that image. That
number is an independent, in-image witness of how many dots were drawn.

So: crop the dialog from a set of screenshots, upscale it for legibility, and
compare the displayed Total Count against the published `total_birds`. If they
agree, `total_birds` is established as "number of dots drawn" and can be used as
the detection ground truth. The dialog is read BY EYE here on purpose — count-OCR
is only ~60-65% reliable on ~10px digits, and this check has to be trustworthy.

Writes dialog crops to results/totalcount_probe/ alongside the published numbers.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.legend import locate_dialog
from scripts.probe_groundtruth import fetch

MANIFEST = "data/cache/manifest.csv"
OUT_DIR = "results/totalcount_probe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-birds", type=int, default=60)
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    per = man.groupby("screenshot_key").agg(
        birds=("total_birds", "sum"), nests=("total_nests", "sum"),
        catsum=("category_sum", "sum"), year=("Year", "first"),
        colony=("ColonyName", "first"),
    ).reset_index()

    sel = per[(per.birds >= 5) & (per.birds <= args.max_birds)]
    sel = sel.sort_values("year").groupby("year", group_keys=False).head(2).head(args.n)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'file':34}{'year':>6}{'birds':>7}{'nests':>7}{'catsum':>8}   crop")
    for _, r in sel.iterrows():
        shot = fetch(r.screenshot_key)
        if shot is None:
            continue
        bbox = locate_dialog(shot)
        if bbox is None:
            print(f"{os.path.basename(r.screenshot_key)[:32]:34}{r.year:>6}"
                  f"{r.birds:>7.0f}{r.nests:>7.0f}{r.catsum:>8.0f}   locate_dialog FAILED")
            continue
        x, y, w, h = bbox
        crop = shot[y:y + h, x:x + w]
        # Upscale so the Total Count digits are legible when reviewed by eye.
        crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        name = f"{r.year}_{os.path.basename(r.screenshot_key)}"
        cv2.imwrite(os.path.join(OUT_DIR, name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print(f"{os.path.basename(r.screenshot_key)[:32]:34}{r.year:>6}"
              f"{r.birds:>7.0f}{r.nests:>7.0f}{r.catsum:>8.0f}   {name}")

    print(f"\nCrops in {OUT_DIR}. Read each dialog's 'Total Count' and compare to "
          f"the 'birds' column above.")


if __name__ == "__main__":
    main()
