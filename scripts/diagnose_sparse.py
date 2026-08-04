"""
Characterise the false ink on the worst sparse frames.

Sparse frames over-detect (median ~2.96x): a truth of ~9-49 dots yields hundreds
of candidates. The report attributes the false ink to red site-label text,
transect lines and residual water. Before tuning anything, look at what the
blobs actually are — per-blob size, elongation, saturation, chroma — and render
an overlay so the failure is inspected spatially, not just by count.

Usage:
    python scripts/diagnose_sparse.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.align import align, warp_to_screenshot, AlignResult
from src.subtract import extract_annotations, dot_candidates, _marker_area

BENCHMARK = "data/cache/benchmark.csv"
PAIR_DIR = "data/fixtures/pairs"
OUT = "results/sparse_probe"

# Worst / representative sparse frames from results/eval_detection.csv.
TARGETS = [
    "10June10Camera1-Card1-0076.jpg",   # truth 9   -> 347
    "18May11Camera2-Card5-0293.jpg",    # truth 36  -> 541
    "15June21Camera2-Card1-03211.jpg",  # truth 13  -> 156
    "24May13Camera1-Card3-0215.jpg",    # truth 30  -> 262
    "18May15Camera1-Card6-00948.jpg",   # truth 11  -> 91
    "30May12Camera2-Card1-0481.jpg",    # truth 33  -> 42  (GOOD, control)
]


def read(key: str):
    path = os.path.join(PAIR_DIR, key.replace("avian_monitoring/", ""))
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    bench = pd.read_csv(BENCHMARK)
    by_name = {os.path.basename(r.screenshot_key): r for r in bench.itertuples()}

    for name in TARGETS:
        r = by_name.get(name)
        if r is None:
            print(f"  {name}: not in benchmark"); continue
        shot, orig = read(r.screenshot_key), read(r.highres_key)
        if shot is None or orig is None:
            print(f"  {name}: image missing"); continue

        res = align(shot, orig)
        if not res.ok:
            print(f"  {name}: alignment failed ({res.reason})"); continue

        sub = extract_annotations(shot, orig, res)
        cands = dot_candidates(sub)

        # Per-blob stats on the RAW ink mask (before dot_candidates filtering).
        n, labels, stats, cents = cv2.connectedComponentsWithStats(sub.mask, 8)
        hsv = cv2.cvtColor(shot, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        areas, elongs, sats, hues = [], [], [], []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < 3:
                continue
            w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
            areas.append(a)
            elongs.append(max(w, h) / max(min(w, h), 1))
            reg = labels == i
            sats.append(float(np.median(sat[reg])))
            hues.append(float(np.median(hsv[:, :, 0][reg])))

        areas = np.array(areas); elongs = np.array(elongs)
        sats = np.array(sats); hues = np.array(hues)
        modal = _marker_area(sub.mask, [(0, 0, 0, 0, 0, int(a), int(a), int(a))
                                        for a in areas]) if areas.size else 0

        print(f"\n=== {name}  band={r.band} truth={r.dots} "
              f"cands={len(cands)} rawblobs={areas.size} ===")
        print(f"  modal marker area (dist-transform): {modal:.1f}")
        if areas.size:
            print(f"  area:  min {areas.min():.0f}  med {np.median(areas):.0f}  "
                  f"p90 {np.percentile(areas,90):.0f}  max {areas.max():.0f}")
            print(f"  elong: med {np.median(elongs):.2f}  p90 "
                  f"{np.percentile(elongs,90):.2f}  max {elongs.max():.2f}  "
                  f"(>4: {(elongs>4).sum()},  >2: {(elongs>2).sum()})")
            print(f"  sat:   med {np.median(sats):.0f}  "
                  f"(<60: {(sats<60).sum()},  >=100: {(sats>=100).sum()})")
            # Blobs that survive dot_candidates' size band, split by elongation.
            in_band = (areas >= 0.35*modal) & (areas <= 3.0*modal)
            print(f"  in size-band [{0.35*modal:.0f},{3.0*modal:.0f}]: "
                  f"{in_band.sum()}  of which elong>2: "
                  f"{(in_band & (elongs>2)).sum()}  sat<60: "
                  f"{(in_band & (sats<60)).sum()}")

        # Overlay: raw ink (red) + accepted candidates (green circles).
        overlay = shot.copy()
        overlay[sub.mask > 0] = (255, 0, 0)
        for (cx, cy, w, h, a) in cands:
            cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(OUT, f"{name}.overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, f"{name}.inkmask.png"), sub.mask)

    print(f"\nwrote overlays to {OUT}/")


if __name__ == "__main__":
    main()
