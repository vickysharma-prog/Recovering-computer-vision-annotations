"""
Measure how large a bird actually is, in original-photograph pixels, per frame.

The survey recorded a point per bird and never an extent, so no box size can be read
off the data. It has to be measured from the imagery, and it has to be measured **per
frame**: the same species is 12px across on one photograph and 23px on another,
because the surveys flew different cameras at different heights over eleven years.

## Why not the obvious proxies

**Spacing between dots does not work.** Measured over the 25 exported frames, the
median nearest-neighbour distance runs from 13.6px on `5745` to 215px on `00825`, a
16x spread, while the birds themselves differ by about 2x. Spacing measures how
crowded a colony is, not how large its birds are. A box built on it would be four
times too large on exactly the sparse frames where a bird is easiest to see.

**A fixed size does not work either.** Birds measure roughly 10-25px across the
frames checked by hand, so the 100px box the export started with is four to eight
times too large everywhere. Every box would then hold several neighbours, and the
model would be asked to learn one bird from a patch containing six.

## What this measures instead

Around each dot, a bird is a small region that does not look like its surroundings —
pale plumage on vegetation, a dark body on sand. So for each isolated dot:

1. Cut a patch centred on the dot from the **clean original**.
2. Take the CIE L channel and measure how far each pixel sits from the patch's own
   median lightness. Contrast against local background, rather than an absolute
   colour, is what survives the change from marsh to sand to open water.
3. Otsu-threshold that contrast, keep the component containing the centre, and take
   the **long side of its minimum-area rectangle**.

Isolated dots are preferred where there are plenty of them: two birds packed together
merge into one component and would be measured as a single large bird. Requiring
isolation everywhere it was possible made the sparse frames unmeasurable, so it is a
preference rather than a rule.

The per-frame answer is the **median** over the dots measured, which is robust to the
components that merge anyway and to the dots that sit on map ink rather than a bird.

## The check that says this is real

On frames dominated by a single species the measurement implies a ground sample
distance, and that can be compared against the species' known body length. Only
species that rest horizontally are used — gulls, terns and skimmers show their full
length from above, while herons and egrets stand upright and foreshorten. Over the 14
frames where such a species holds at least half the resolved dots:

    median 1.27 cm/px      range 0.90 - 2.92 cm/px

That is the range aerial surveys of this kind fly, and it agrees with the cameras: the
EXIF records focal lengths from 28mm to 300mm and pixel pitches from 4.4 to 6.6 um, so
    GSD = altitude * pitch / focal
predicts exactly the tenfold spread in bird size that the measurement finds. The
method is recovering a physical quantity rather than fitting one.

Usage:
    est = frame_bird_size(original_rgb, points_xy)
    box = box_from_size(est.median_px)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

# A bird smaller than this is noise and larger than this is a merged pair or a patch
# of map ink. Both bounds sit well outside the 10-25px range measured by hand, so they
# reject failures rather than shaping the answer.
MIN_PX = 4.0
MAX_PX = 80.0

# Longest-to-shortest side a single bird can plausibly show from above.
MAX_ASPECT = 3.0

# Patch side, in original pixels. Wide enough to hold the largest plausible bird with
# background around it for the median to describe, narrow enough that a second bird
# usually falls outside.
PATCH = 60

# A dot at least this far from its nearest neighbour is treated as isolated.
# Isolation is preferred, not required. Measured on the 25 exported frames, insisting
# on it costs more than the merging it avoids: on `0483` it cut 234 dots to 7 and the
# frame became unmeasurable, while using every dot gives a median of 13.5px from about
# ninety readings that agree with each other. A merged pair lands above MAX_PX or in
# the upper tail, and a median over dozens of dots is not moved by either.
ISOLATED_PX = 30.0
ISOLATED_ENOUGH = 25          # prefer isolated dots only when this many exist


# Below this many measurable dots a frame's median is not worth trusting. Five is low
# because the sparse frames genuinely hold few birds: `00825` has seven in total.
MIN_DOTS = 5


@dataclass(frozen=True)
class SizeEstimate:
    """A frame's bird size, with enough context to judge whether to believe it."""

    median_px: Optional[float]     # None when too few dots could be measured
    p25_px: Optional[float]
    p75_px: Optional[float]
    n_measured: int
    n_offered: int
    isolated_only: bool            # False when the frame had too few isolated dots

    @property
    def ok(self) -> bool:
        return self.median_px is not None


def _extent_at(patch: np.ndarray) -> Optional[float]:
    """Equivalent diameter of the object under the patch's centre, or None."""
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    # Distance from the patch's own background, so a pale bird on grass and a dark
    # bird on sand are both "unlike their surroundings".
    contrast = np.abs(lightness - float(np.median(lightness)))
    if contrast.max() <= 0:
        return None

    # Otsu, not a fixed percentile. A percentile assumes the bird occupies a known
    # share of the patch, and it does not: the share changes with the frame's scale
    # and with how many neighbours fall inside. Measured against discs of known size
    # on noisy ground, a p88 cut read a 14px disc as 21px because it swept in the
    # loudest eighth of the background with it. Otsu splits the patch where its own
    # two populations separate, so it follows the scene instead of assuming it.
    scaled = np.clip(contrast / contrast.max() * 255.0, 0, 255).astype(np.uint8)
    _t, mask = cv2.threshold(scaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None

    cy, cx = patch.shape[0] // 2, patch.shape[1] // 2
    idx = int(labels[cy, cx])
    if idx == 0:
        # The dot sits just off the bird, which placement error alone can cause.
        # Take the nearest component instead, if one is close.
        best, best_d = 0, 1e9
        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            d = np.hypot(x + w / 2 - cx, y + h / 2 - cy)
            if d < best_d:
                best, best_d = i, d
        if best == 0 or best_d > 8:
            return None
        idx = best

    if stats[idx, cv2.CC_STAT_AREA] <= 0:
        return None

    # The LONG side of the minimum-area rectangle, not the equivalent diameter.
    #
    # An equivalent diameter is the width of a circle with the bird's area, and a
    # bird is not a circle. On the frames whose resolution resolves a bird shape the
    # two disagree badly, and the box built from the smaller one cuts the bird in
    # half:
    #
    #     frame   equiv diameter   long axis
    #     5745         10.6           11.7     agree; at 3.3cm/px a gull is a blob
    #     0027         13.9           24.5
    #     0406         15.4           43.2
    #
    # Minimum-area rather than axis-aligned, so the answer does not depend on which
    # way the bird happens to be lying.
    comp = (labels == idx).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (_centre, (rw, rh), _angle) = cv2.minAreaRect(cnts[0])
    long_side, short_side = max(rw, rh), min(rw, rh)
    if short_side <= 0:
        return None

    # Reject anything too elongated to be one bird. A gull or tern seen from above is
    # at most about three times as long as it is wide, bill and tail included; beyond
    # that the component is a row of birds that touched, or a strand of map ink.
    #
    # This matters far more for the long axis than it did for the equivalent
    # diameter: two birds merging doubles the long side but multiplies the
    # area-equivalent circle by only root two. Without the cut, `0406` reads a long
    # axis of 38px against an equivalent diameter of 15px, a ratio of 2.5 where a
    # single 2:1 bird should give about 1.4.
    if long_side / short_side > MAX_ASPECT:
        return None
    return float(long_side)


def frame_bird_size(original_rgb: np.ndarray,
                    points_xy: Sequence[Sequence[float]],
                    patch: int = PATCH) -> SizeEstimate:
    """Median bird extent for one frame, in original pixels."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return SizeEstimate(None, None, None, 0, 0, False)

    isolated_only = False
    if len(pts) > 2:
        from scipy.spatial import cKDTree
        nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
        keep = nn >= ISOLATED_PX
        # Prefer isolated dots only when there are plenty of them. Requiring isolation
        # wherever it was possible made the sparse and medium frames unmeasurable,
        # which is the opposite of the intent: those are the frames whose birds are
        # easiest to see.
        if keep.sum() >= ISOLATED_ENOUGH:
            pts, isolated_only = pts[keep], True

    h, w = original_rgb.shape[:2]
    half = patch // 2
    sizes = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if not (half <= xi < w - half and half <= yi < h - half):
            continue
        d = _extent_at(original_rgb[yi - half:yi + half, xi - half:xi + half])
        if d is not None and MIN_PX <= d <= MAX_PX:
            sizes.append(d)

    if len(sizes) < MIN_DOTS:
        return SizeEstimate(None, None, None, len(sizes), len(pts), isolated_only)

    a = np.array(sizes)
    return SizeEstimate(float(np.median(a)), float(np.percentile(a, 25)),
                        float(np.percentile(a, 75)), len(sizes), len(pts),
                        isolated_only)


def box_from_size(median_px: float, k: float = 1.3,
                  lo: int = 16, hi: int = 120) -> int:
    """Box side for a bird whose long axis measures `median_px`.

    `k` leaves a margin rather than cropping to the silhouette: a detector trained on
    boxes cut exactly to the object has nothing to place the object against. It is
    smaller than it would be for an equivalent diameter, because the long axis
    already spans the bird's longest dimension — the margin is context, not a
    correction for the wrong axis.

    The bounds are guards on a failed measurement, not tuning. The smallest frame
    measured, `5745` at 11.7px, sits just under the floor; nothing comes near the
    ceiling.
    """
    return int(round(min(max(median_px * k, lo), hi)))
