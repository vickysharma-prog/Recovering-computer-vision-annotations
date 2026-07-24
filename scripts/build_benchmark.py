"""
Select and download a stratified benchmark of screenshot / clean-original pairs.

Usage:
    python scripts/build_benchmark.py [--per-cell 3] [--dry-run]

Why stratified
--------------
All prior numbers came from four images, which is how we ended up tuning colour
thresholds that overfit. Annotation symbology changes across survey years, and the
corpus is heavily skewed by density (median 61 dots/photo, max 7346), so the
benchmark samples every year crossed with a density band, and deliberately
includes the dense tail — dense colonies are where detection is weakest.

Ground truth is `category_sum` (the sum of the per-dot-type columns), which was
verified against the counting tool's own "Total Count" field. It is NOT
`total_birds`, an ecological metric that excludes chicks and undercounts dots by
up to 57% on a single image. See scripts/probe_totalcount.py.

Writes data/cache/benchmark.csv and downloads pairs under data/fixtures/pairs/.
Originals are ~7MB each, so mind the disk: cost is printed before downloading.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.probe_groundtruth import fetch

MANIFEST = "data/cache/manifest.csv"
OUT = "data/cache/benchmark.csv"

# Dot-count bands. "dense" is where the pipeline historically fails, so it is
# sampled on equal footing rather than in proportion to its rarity.
BANDS = [("sparse", 5, 50), ("medium", 51, 300), ("dense", 301, 10 ** 9)]


def per_screenshot(man: pd.DataFrame) -> pd.DataFrame:
    return man.groupby(["screenshot_key", "highres_key"]).agg(
        dots=("category_sum", "sum"),
        birds=("total_birds", "sum"),
        nests=("total_nests", "sum"),
        species=("SpeciesCode", "nunique"),
        year=("Year", "first"),
        colony=("ColonyName", "first"),
        region=("GeoRegion", "first"),
    ).reset_index()


def select(per: pd.DataFrame, per_cell: int, seed: int = 0) -> pd.DataFrame:
    picks = []
    for year in sorted(per.year.unique()):
        for band, lo, hi in BANDS:
            cell = per[(per.year == year) & (per.dots >= lo) & (per.dots <= hi)]
            if cell.empty:
                continue
            # Spread across colonies so a single big colony cannot dominate a cell.
            take = (cell.sample(frac=1.0, random_state=seed)
                        .drop_duplicates("colony").head(per_cell))
            if len(take) < per_cell:                       # not enough colonies
                extra = cell[~cell.screenshot_key.isin(take.screenshot_key)]
                take = pd.concat([take, extra.sample(
                    min(per_cell - len(take), len(extra)), random_state=seed)])
            picks.append(take.assign(band=band))
    return pd.concat(picks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=3,
                    help="images per (year x density band) cell")
    ap.add_argument("--dry-run", action="store_true", help="select but do not download")
    args = ap.parse_args()

    if not os.path.exists(MANIFEST):
        sys.exit("run scripts/build_manifest.py first")

    per = per_screenshot(pd.read_csv(MANIFEST, low_memory=False))
    sel = select(per, args.per_cell)

    print(sel.groupby(["year", "band"]).agg(n=("dots", "size"),
                                            dots=("dots", "median")).to_string())
    print(f"\nselected {len(sel)} pairs   dots: median {sel.dots.median():.0f}  "
          f"min {sel.dots.min():.0f}  max {sel.dots.max():.0f}")
    print(f"years {sel.year.nunique()}  colonies {sel.colony.nunique()}  "
          f"regions {sel.region.nunique()}")
    print(f"estimated download ~{len(sel) * 7.5:.0f} MB (originals dominate)")

    os.makedirs("data/cache", exist_ok=True)
    sel.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    if args.dry_run:
        return
    ok = 0
    for i, r in enumerate(sel.itertuples(), 1):
        got_shot = fetch(r.screenshot_key) is not None
        got_orig = fetch(r.highres_key) is not None
        ok += got_shot and got_orig
        if i % 10 == 0 or i == len(sel):
            print(f"  fetched {i}/{len(sel)}  ({ok} complete pairs)")
    print(f"\n{ok}/{len(sel)} pairs available under data/fixtures/pairs/")


if __name__ == "__main__":
    main()
