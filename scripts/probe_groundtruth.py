"""
Phase 0 gate: work out what a survey count actually MEANS in terms of dots drawn.

Usage:
    python scripts/probe_groundtruth.py [--n 6] [--max-birds 40]

Why this exists
---------------
Every detection metric we report is "detected vs ground truth", so we must know
which published quantity corresponds to the dots a dotter actually clicked onto
the screenshot. That is currently unknown:

  * `total_birds` != sum of the per-category columns on ~24.5% of screenshots.
  * Our four local benchmark images do not map cleanly onto rows in the survey
    data (D's trusted "true = 636" has no exact match; nearest are 631/629/643).

So we measure it instead of assuming it.

Method
------
Pick SPARSE screenshots (few birds), where dots are individually unambiguous.
For each: download the screenshot and its paired clean original, align them with
SIFT + RANSAC, warp the original into the screenshot frame, and difference. What
remains is annotation ink. Count the blobs, and write an overlay image so the
count can be checked BY EYE — that visual check is the independent anchor here,
which is what keeps this from being circular (we are not using the survey counts
to decide what a dot is).

Then compare the eyeballed dot count against total_birds, total_nests and the
category sum, and see which one tracks.

Outputs overlays to results/groundtruth_probe/ and prints a comparison table.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUCKET = "https://twi-aviandata.s3.amazonaws.com/"
MANIFEST = "data/cache/manifest.csv"
PAIR_DIR = "data/fixtures/pairs"
OUT_DIR = "results/groundtruth_probe"


def fetch(key: str) -> np.ndarray | None:
    """Download an S3 object by key, caching under data/fixtures/pairs/."""
    local = os.path.join(PAIR_DIR, key.replace("avian_monitoring/", ""))
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        url = BUCKET + urllib.parse.quote(key)
        try:
            data = urllib.request.urlopen(url, timeout=300).read()
        except Exception as exc:                                   # noqa: BLE001
            print(f"    fetch failed: {type(exc).__name__} {exc}")
            return None
        with open(local, "wb") as f:
            f.write(data)
    img = cv2.imread(local)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def align(shot: np.ndarray, orig: np.ndarray):
    """SIFT + RANSAC homography mapping shot -> downscaled orig. None if weak."""
    scale = 1600 / orig.shape[1]
    small = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(shot, cv2.COLOR_RGB2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, None
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or int(mask.sum()) < 8:
        return None, None
    return H, small


def annotation_mask(shot: np.ndarray, small: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Difference the aligned original against the screenshot -> ink mask."""
    warped = cv2.warpPerspective(small, np.linalg.inv(H), (shot.shape[1], shot.shape[0]))
    covered = cv2.warpPerspective(np.full(small.shape[:2], 255, np.uint8),
                                  np.linalg.inv(H), (shot.shape[1], shot.shape[0]))

    # Match exposure before differencing: the screenshot is a ~4x downscaled,
    # re-encoded render, so channel gain/offset drift from the original.
    a = cv2.cvtColor(shot, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    valid = covered > 0
    for c in range(3):
        av, bv = a[..., c][valid], b[..., c][valid]
        if bv.std() > 1e-3:
            b[..., c] = (b[..., c] - bv.mean()) * (av.std() / bv.std()) + av.mean()

    diff = np.linalg.norm(a - b, axis=-1)
    diff[~valid] = 0
    thr = float(np.percentile(diff[valid], 99.0))
    mask = (diff > max(thr, 18.0)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def count_dots(mask: np.ndarray):
    """Blobs that look like annotation dots (small, roughly round)."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # Reject the dialog and other UI slabs: real dots are small and compact.
        if area < 4 or area > 400 or w > 40 or h > 40:
            continue
        if max(w, h) / max(1, min(w, h)) > 4:
            continue
        keep.append((cents[i][0], cents[i][1], area))
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-birds", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(MANIFEST):
        sys.exit("run scripts/build_manifest.py first")
    man = pd.read_csv(MANIFEST)

    per = man.groupby(["screenshot_key", "highres_key"]).agg(
        birds=("total_birds", "sum"), nests=("total_nests", "sum"),
        catsum=("category_sum", "sum"), year=("Year", "first"),
        colony=("ColonyName", "first"), species=("SpeciesCode", "nunique"),
    ).reset_index()

    # Sparse images only: unambiguous dots, and small numbers make a
    # discrepancy between the candidate quantities obvious.
    sel = per[(per.birds >= 5) & (per.birds <= args.max_birds)]
    sel = sel.sort_values("year").groupby("year", group_keys=False).head(2).head(args.n)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for _, r in sel.iterrows():
        name = os.path.basename(r.screenshot_key)
        print(f"\n{r.year} {r.colony} — {name}")
        shot, orig = fetch(r.screenshot_key), fetch(r.highres_key)
        if shot is None or orig is None:
            continue
        H, small = align(shot, orig)
        if H is None:
            print("    alignment FAILED")
            rows.append(dict(name=name, year=r.year, colony=r.colony, detected=None,
                             birds=r.birds, nests=r.nests, catsum=r.catsum))
            continue
        dots = count_dots(annotation_mask(shot, small, H))

        vis = cv2.cvtColor(shot, cv2.COLOR_RGB2BGR)
        for cx, cy, _ in dots:
            cv2.circle(vis, (int(cx), int(cy)), 9, (0, 255, 255), 2)
        out = os.path.join(OUT_DIR, f"{r.year}_{name}")
        cv2.imwrite(out, vis)
        print(f"    blobs={len(dots):4d}   birds={r.birds:.0f} nests={r.nests:.0f} "
              f"catsum={r.catsum:.0f}   -> {out}")
        rows.append(dict(name=name, year=r.year, colony=r.colony, detected=len(dots),
                         birds=r.birds, nests=r.nests, catsum=r.catsum))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    ok = df.dropna(subset=["detected"])
    if len(ok):
        print("\nmean |detected - X| :  birds %.1f   nests %.1f   catsum %.1f" % (
            (ok.detected - ok.birds).abs().mean(),
            (ok.detected - ok.nests).abs().mean(),
            (ok.detected - ok.catsum).abs().mean()))
    print("\nNOW LOOK AT THE OVERLAYS in", OUT_DIR,
          "\nThe blob count is only trustworthy if the circles land on real dots.")


if __name__ == "__main__":
    main()
