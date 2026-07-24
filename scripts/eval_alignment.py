"""
Measure how often screenshot <-> original registration actually succeeds.

Usage:
    python scripts/eval_alignment.py [--limit N]

Alignment is the ceiling on everything difference-based detection can deliver:
any pair that fails to register has to fall back to colour thresholds. So the
success rate is reported honestly, broken out by survey year, density band and
region, rather than quoted as a single headline number.

Pairs that have not been downloaded yet are skipped, so this can be run against a
partially fetched benchmark.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import align

BENCHMARK = "data/cache/benchmark.csv"
PAIR_DIR = "data/fixtures/pairs"


def local(key: str) -> str:
    return os.path.join(PAIR_DIR, key.replace("avian_monitoring/", ""))


def read(key: str):
    path = local(key)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(BENCHMARK):
        sys.exit("run scripts/build_benchmark.py first")
    bench = pd.read_csv(BENCHMARK)
    if args.limit:
        bench = bench.head(args.limit)

    rows, skipped = [], 0
    for r in bench.itertuples():
        shot, orig = read(r.screenshot_key), read(r.highres_key)
        if shot is None or orig is None:
            skipped += 1
            continue
        res = align(shot, orig)
        rows.append(dict(
            name=os.path.basename(r.screenshot_key), year=r.year, band=r.band,
            region=r.region, colony=r.colony, dots=r.dots,
            ok=res.ok, model=res.model, matches=res.matches, inliers=res.inliers,
            inlier_frac=round(res.inlier_frac, 3),
            reproj=round(res.reproj_err, 2) if res.ok else None,
            reason=res.reason))
        mark = "ok " if res.ok else "FAIL"
        print(f"  {mark} {os.path.basename(r.screenshot_key)[:34]:36}"
              f"{res.model:11}{res.inliers:>4}/{res.matches:<4}"
              f"{(f'{res.reproj_err:.2f}px' if res.ok else res.reason)}")

    if not rows:
        sys.exit(f"no downloaded pairs found (skipped {skipped})")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print(f"pairs evaluated {len(df)}   (skipped {skipped} not yet downloaded)")
    print(f"ALIGNMENT SUCCESS: {df.ok.mean():.1%}  ({df.ok.sum()}/{len(df)})")
    ok = df[df.ok]
    if len(ok):
        print(f"  model chosen : {dict(ok.model.value_counts())}")
        print(f"  reproj err   : median {ok.reproj.median():.2f}px  "
              f"max {ok.reproj.max():.2f}px")
        print(f"  inlier frac  : median {ok.inlier_frac.median():.0%}")

    for key in ("year", "band", "region"):
        g = df.groupby(key).ok.agg(["mean", "size"])
        print(f"\nby {key}:")
        for k, row in g.iterrows():
            print(f"  {str(k)[:28]:30}{row['mean']:>7.0%}  (n={row['size']:.0f})")

    bad = df[~df.ok]
    if len(bad):
        print("\nfailures:")
        for b in bad.itertuples():
            print(f"  {b.name[:38]:40}{b.year}  {b.band:7}{b.reason}")

    out = "results/eval_alignment.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
