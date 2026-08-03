"""
Build a local ground-truth manifest from the published survey data.

Usage:
    python scripts/build_manifest.py [--refresh]

Downloads (and caches) `avianData20102021.csv.gz` from the public TWI bucket and
writes `data/cache/manifest.csv` — one row per (screenshot, species) carrying the
per-category dot counts plus the S3 keys of the screenshot and its paired clean
high-resolution original.

IMPORTANT — this manifest is EVALUATION ONLY.
`src/` must never import it. The pipeline is required to work from the image alone
(the screenshot and its paired original); these counts exist to measure the
pipeline, not to feed it. Using them as a pipeline input would make any per-class
count metric circular.

Note also that "what a CSV count means in terms of dots drawn on the image" is not
yet established — `total_birds` does not equal the sum of the category columns on
~12% of rows. See scripts/probe_groundtruth.py, which settles that question.
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUCKET = "https://twi-aviandata.s3.amazonaws.com/"
SOURCE = BUCKET + "avian_monitoring/dotting_information/processed_data/avianData20102021.csv.gz"

CACHE_DIR = "data/cache"
RAW_PATH = os.path.join(CACHE_DIR, "avianData20102021.csv.gz")
OUT_PATH = os.path.join(CACHE_DIR, "manifest.csv")

# The per-dot-type count columns: the quantities a dotter incremented while
# clicking. Their sum is the number of dots drawn on the image — verified against
# the counting tool's own "Total Count" field (see scripts/probe_totalcount.py).
#
# NOTE the four separate chick-like columns. `ChicksNestlings` and
# `Chicks/Nestlings` are DIFFERENT columns and both carry data; omitting either
# silently undercounts. That is not hypothetical — dropping `ChicksNestlings`
# made 17June13Camera1-Card1-0048 read 11 dots when the dialog says 18.
CATEGORY_COLS = [
    "WBN", "Site", "Brood", "ChickNest", "ChickNestwithoutAdult", "PBN",
    "Chicks/Nestlings", "ChicksNestlings", "RoostingBirds", "RoostingAdults",
    "RoostingImmatures", "OtherAdultsInColony", "OtherImmInColony", "AbandNest",
    "EmptyNest", "Territory", "UnknownAge", "OtherBirds",
]

KEEP_META = [
    "Year", "Date", "ColonyName", "GeoRegion", "State", "SpeciesCode",
    "Dotter", "PhotoNumber", "CameraNumber", "CardNumber",
    "total_birds", "total_nests",
]


def fetch_raw(refresh: bool = False) -> bytes:
    """Download the survey csv.gz, caching it locally."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH) and not refresh:
        print(f"using cached {RAW_PATH}")
        return open(RAW_PATH, "rb").read()
    print(f"downloading {SOURCE} ...")
    raw = urllib.request.urlopen(SOURCE, timeout=300).read()
    with open(RAW_PATH, "wb") as f:
        f.write(raw)
    print(f"cached {len(raw):,} bytes -> {RAW_PATH}")
    return raw


def build(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False)

    # Only rows that name both images are usable as a benchmark pair.
    df = df.dropna(subset=["screenshot_new", "HighResImage_new"]).copy()

    cats = [c for c in CATEGORY_COLS if c in df.columns]
    meta = [c for c in KEEP_META if c in df.columns]
    out = df[meta + cats].copy()
    out[cats] = out[cats].fillna(0)

    out.insert(0, "screenshot_key", df["screenshot_new"].values)
    out.insert(1, "highres_key", df["HighResImage_new"].values)
    # Sum of the per-dot-type columns. Deliberately kept SEPARATE from
    # total_birds: the two disagree, and which one tracks drawn dots is exactly
    # the open question.
    out["category_sum"] = out[cats].sum(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the source csv")
    args = ap.parse_args()

    out = build(fetch_raw(args.refresh))
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    per_shot = out.groupby("screenshot_key").agg(
        species=("SpeciesCode", "nunique"),
        birds=("total_birds", "sum"),
        nests=("total_nests", "sum"),
        catsum=("category_sum", "sum"),
    )
    print(f"\nwrote {OUT_PATH}  ({len(out):,} rows)")
    print(f"  unique screenshots : {out.screenshot_key.nunique():,}")
    print(f"  years              : {sorted(out.Year.unique())}")
    print(f"  colonies           : {out.ColonyName.nunique()}")
    print(f"  species codes      : {out.SpeciesCode.nunique()}")
    print(f"  birds/photo        : median {per_shot.birds.median():.0f}  "
          f"max {per_shot.birds.max():.0f}")
    disagree = (per_shot.catsum != per_shot.birds).mean()
    print(f"  category_sum != total_birds on {disagree:.1%} of screenshots "
          f"(unresolved — see scripts/probe_groundtruth.py)")


if __name__ == "__main__":
    main()
