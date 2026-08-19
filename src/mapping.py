"""
Map detected dots from screenshot coordinates onto the clean original photograph.

Detection and classification both work in the screenshot's own pixel grid, because
that is where the counting tool baked the dots. Nothing downstream can use that: a
model trains on the originals, which are roughly three times larger and do not carry
the dialog, the desktop, or the window chrome. This module is the step between.

`align.align()` already recovers the transform and `data/cache/align_cache.json`
already holds it for the benchmark pairs, so there is no fitting here — only the
arithmetic, and one unit conversion that is easy to get wrong.

## The transform does not land on the original

`AlignResult.H` maps screenshot pixels to **work-scale** original pixels, where
`work = original * res.scale`. The original is downscaled before SIFT runs, because
matching against a 3168x4752 image is slow and buys nothing. `warp_to_screenshot`
confirms the direction by inverting `H`.

So the full-resolution coordinate is:

    orig_xy = perspectiveTransform(shot_xy, H) / res.scale

Dropping the divide is the trap. It returns coordinates that are internally
consistent, plausibly sized, and a factor of `scale` too small — on
`17May10Camera2-Card1-5745` (scale 0.5319) roughly half. Nothing downstream would
raise, and every box would sit in the wrong part of the photograph.

## Out of bounds is a real answer, not an error

The screenshot is wider than the original in work units: on that same frame
`1855 * H[0,0] / scale = 3710` against an original width of 3008. The surplus is
desktop and window chrome, which the original has no counterpart for. A dot landing
there is kept with `in_bounds=False` rather than dropped, so the export can report
how many were lost and why instead of quietly shrinking.

Usage:
    res = align(screenshot_rgb, original_rgb)          # or the cached transform
    if res.ok:
        mapped = map_dots(dots, res, original_rgb.shape)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


# ─────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MappedDot:
    """One detected dot, placed on the original photograph.

    Both coordinate pairs are kept. The screenshot pair is what every earlier stage
    and every hand label is expressed in, so keeping it is what lets an exported row
    be traced back to the frame it came from.
    """

    index: int            # position in the input list, so the source dot can be rejoined
    x: float              # original-photograph pixels
    y: float
    shot_x: float         # screenshot pixels, as detected
    shot_y: float
    in_bounds: bool       # does (x, y) fall inside the original's extent


# ─────────────────────────────────────────────────────
# MAPPING
# ─────────────────────────────────────────────────────

def to_original(points_xy: Sequence[Sequence[float]], res: Any) -> np.ndarray:
    """Map screenshot points to full-resolution original points.

    `res` is an `align.AlignResult`. A rejected alignment raises rather than returning
    a silently wrong answer, mirroring `align.warp_to_screenshot`: at corpus scale a
    bad transform applied to 18k images is far worse than a refusal.

    Returns an (N, 2) float64 array. An empty input returns an empty (0, 2) array.
    """
    if not getattr(res, "ok", False) or res.H is None:
        raise ValueError("cannot map with a rejected alignment")
    if res.scale <= 0:
        raise ValueError(f"alignment scale must be positive, got {res.scale}")

    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    work = cv2.perspectiveTransform(pts, np.asarray(res.H, dtype=np.float64))
    # work-scale -> full resolution. See the module docstring: this divide is the
    # whole reason this function exists rather than a bare perspectiveTransform call.
    return work.reshape(-1, 2) / res.scale


def map_dots(dots: Iterable[Any], res: Any,
             original_shape: Sequence[int]) -> list[MappedDot]:
    """Map detected dots onto the original, flagging any that land outside it.

    `dots` is any sequence of objects carrying `cx` and `cy` — `classify.AerialDot`
    in the live pipeline. `original_shape` is the original image's `.shape`; only the
    first two entries are read, so a full `(h, w, 3)` is fine.
    """
    dots = list(dots)
    if not dots:
        return []

    h, w = int(original_shape[0]), int(original_shape[1])
    mapped = to_original([(d.cx, d.cy) for d in dots], res)

    return [
        MappedDot(
            index=i, x=float(x), y=float(y),
            shot_x=float(d.cx), shot_y=float(d.cy),
            in_bounds=bool(0 <= x < w and 0 <= y < h),
        )
        for i, (d, (x, y)) in enumerate(zip(dots, mapped))
    ]
