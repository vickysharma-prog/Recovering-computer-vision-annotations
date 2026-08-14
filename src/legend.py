"""
Dialog legend parsing for bird annotation recovery.

The point-counting GUI renders a legend table inside the dialog box:
each row shows a colored marker glyph, a class name, and a count, all on
the same line. The marker's SHAPE and COLOR together identify the class —
color alone is ambiguous (e.g. a red circle = "LAGU Stand" but a red
square = "LAGU Roost"; a dark-red circle = "BRPE Ad").

This module extracts, per image, the ordered list of legend markers with
their shape + color. Those become per-image templates used to classify
dots in the aerial region, so categories that share a color but differ in
shape are kept separate.

Pipeline position: runs on the dialog region from ScreenshotDecomposer.
Input: dialog region (RGB numpy array)
Output: list[LegendEntry]

Design goal (per mentor): the detector should work from the screenshot
alone, without the annotation CSV. The CSV is used only to validate the
recovered mapping.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        full = yaml.safe_load(f)
    return full.get("legend", {})


_CONFIG = _load_config()

# Defaults keep the module usable even if config.yaml has no `legend:` block.
_BLOB_SAT_MIN = _CONFIG.get("blob_sat_min", 90)
_SAT_MIN = _CONFIG.get("marker_sat_min", 70)
_VAL_MIN = _CONFIG.get("marker_val_min", 55)
_BLOB_AREA_MIN = _CONFIG.get("blob_area_min", 5)
_BLOB_AREA_MAX = _CONFIG.get("blob_area_max", 70)
_MARKER_BAND_PAD = _CONFIG.get("marker_band_pad", 5)
_PITCH_MIN = _CONFIG.get("row_pitch_min", 6.0)
_PITCH_MAX = _CONFIG.get("row_pitch_max", 12.0)
_FG_GREY_DELTA = _CONFIG.get("fg_grey_delta", 28)
_FILL_FILLED = _CONFIG.get("fill_filled", 0.55)
_LADDER_MIN = _CONFIG.get("ladder_min_rows", 3)
_DIALOG_MAX_AREA = _CONFIG.get("dialog_max_area_frac", 0.40)
_TEXT_MIN_COLS = _CONFIG.get("row_text_min_cols", 10)
_TEXT_MIN_RUNS = _CONFIG.get("row_text_min_runs", 3)
# Shortest cleaned token `_find_species` will try to match. Species codes are four
# letters, but OCR routinely drops the leading one; at 4 such a token never reaches
# the matcher. SPECIES_MIN_TOKEN=4 restores the old gate for A/B.
_SPECIES_MIN_TOKEN = int(os.environ.get("SPECIES_MIN_TOKEN", "3"))
_SQUARE_EXTENT = _CONFIG.get("square_extent", 0.72)


# ─────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────

@dataclass
class LegendEntry:
    """
    One parsed legend row.

    Attributes:
        row: Vertical order index (0 = top row).
        cy: Marker center y in dialog coords.
        cx: Marker center x in dialog coords.
        shape: 'circle', 'square', 'star', 'triangle', 'ring', 'plus',
               or 'unknown' (best-effort label; matching uses `template`).
        color: HSV color name ('red', 'yellow', ...) or 'grey'.
        hue: Circular-mean hue [0,180] of marker foreground, or None for grey.
        marker: Cropped marker glyph (RGB), for visualization.
        template: Canonical (T x T) float glyph mask in [0,1], centered and
                  scale-normalized, used to match aerial dots by shape.
        class_name: Class label read from the dialog text (OCR), or None.
        species: Species code parsed from class_name (e.g. 'BRPE'), or None.
        category: Category parsed from class_name (e.g. 'Site'), or None.
        count: Per-class count read from the dialog Count column, or None.
    """
    row: int
    cy: float
    cx: float
    shape: str
    color: str
    hue: Optional[float]
    marker: np.ndarray
    template: np.ndarray
    class_name: Optional[str] = None
    species: Optional[str] = None
    category: Optional[str] = None
    count: Optional[int] = None

    def key(self) -> str:
        """Stable (shape, color) identity used to match aerial dots."""
        return f"{self.color}:{self.shape}"

    def __repr__(self) -> str:
        return (
            f"LegendEntry(row={self.row}, {self.color}/{self.shape}, "
            f"cy={self.cy:.0f}, cx={self.cx:.0f})"
        )


# ─────────────────────────────────────────────────────
# COLOR
# ─────────────────────────────────────────────────────

# Hue ranges mirror detect.py's color_bins (OpenCV hue 0-180).
_HUE_NAMES: list[tuple[str, list[tuple[int, int]]]] = [
    ("red", [(0, 10), (160, 180)]),
    ("orange", [(11, 22)]),
    ("yellow", [(23, 33)]),
    ("green", [(34, 85)]),
    ("cyan", [(86, 100)]),
    ("blue", [(101, 130)]),
    ("magenta", [(131, 159)]),
]


def _circular_mean_hue(hues: np.ndarray) -> float:
    """Circular mean of OpenCV hues (0-180); handles red's wrap-around."""
    if hues.size == 0:
        return 0.0
    rad = hues.astype(float) * (np.pi / 90.0)
    mean = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    return float((mean * 90.0 / np.pi) % 180.0)


def _name_hue(hue: float) -> str:
    for name, ranges in _HUE_NAMES:
        for lo, hi in ranges:
            if lo <= hue <= hi:
                return name
    return "red" if hue >= 160 or hue <= 10 else "unknown"


# ─────────────────────────────────────────────────────
# GRID DETECTION
# ─────────────────────────────────────────────────────

def _raw_components(mask: np.ndarray) -> list[tuple[float, float, int, int]]:
    """Connected components as (cy, cx, area, size) where size=max(bw,bh)."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) / max(min(bw, bh), 1) > 4:
            continue
        out.append((float(cents[i][1]), float(cents[i][0]), area, max(bw, bh)))
    return out


def _marker_blobs(hsv: np.ndarray) -> tuple[list[tuple[float, float, int]], float]:
    """
    Colored marker blobs (cy, cx, area) plus the estimated marker size.

    Scale-adaptive: thumbnail markers are ~6 px, full-res ~15 px. We first
    estimate marker size from the larger (non-fragmented) blobs, then close
    the mask with a kernel scaled to that size so asterisk/plus markers — whose
    thin arms otherwise split into several components — merge into one blob.
    """
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    base = ((s > _BLOB_SAT_MIN) & (v > _VAL_MIN)).astype(np.uint8) * 255

    # Pass 1: rough size from filled markers (they don't fragment).
    rough = _raw_components(
        cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    )
    if len(rough) < 3:
        return [], 0.0
    sizes = np.array([c[3] for c in rough])
    msize = float(np.percentile(sizes, 70))  # typical full marker size

    # Pass 2: close with a kernel ~half the marker size to merge fragments.
    k = max(2, int(round(msize * 0.5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    merged = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel)
    comps = _raw_components(merged)

    # Keep uniform, marker-sized blobs (drop tiny noise and huge UI chrome).
    lo, hi = msize * 0.45, msize * 2.2
    blobs = [
        (cy, cx, area) for (cy, cx, area, sz) in comps if lo <= sz <= hi
    ]
    return blobs, msize


def _fit_grid(
    blobs: list[tuple[float, float, int]], marker_size: float, dh: int
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate (marker_x, pitch, phase) from colored marker blobs.

    Pitch is derived from the data (the dominant single-row gap), so it works
    at any resolution. Returns marker column x, row pitch, and a reference y.
    """
    if len(blobs) < 3:
        return None, None, None

    band_pad = max(_MARKER_BAND_PAD, marker_size * 0.8)

    # Marker column = the densest vertical x-cluster, NOT the median: legends
    # often have stray colored pixels in the Name/Count columns that would drag
    # a median off the true marker column.
    xs = np.array([b[1] for b in blobs])
    best_x, best_n = float(np.median(xs)), 0
    for cand in xs:
        n = int((np.abs(xs - cand) <= band_pad).sum())
        if n > best_n:
            best_n, best_x = n, float(cand)
    in_col = xs[np.abs(xs - best_x) <= band_pad]
    marker_x = float(np.median(in_col))

    band = [b for b in blobs if abs(b[1] - marker_x) <= band_pad]
    ys = np.sort(np.array([b[0] for b in band]))
    if ys.size < 3:
        return marker_x, None, None

    # Row pitch = the single-row gap. Among adjacent colored markers, the
    # smallest real gap is one row; larger gaps are integer multiples (grey
    # rows in between). A low percentile of valid gaps recovers it robustly,
    # floored at the marker size (rows cannot overlap). A short comb-fit then
    # refines the estimate against all rows.
    diffs = np.diff(ys)
    valid = diffs[diffs >= marker_size * 0.6]
    if valid.size == 0:
        return marker_x, None, float(ys[0])
    anchor = max(marker_size * 0.85, float(np.percentile(valid, 12)))

    best_p, best_err = anchor, None
    for p in np.arange(anchor * 0.8, anchor * 1.25, 0.2):
        k = np.round((ys - ys[0]) / p)
        pred = ys[0] + k * p
        err = float(np.mean(np.abs(ys - pred)))
        if best_err is None or err < best_err:
            best_err, best_p = err, float(p)
    if best_p < 3:
        return marker_x, None, float(ys[0])
    return marker_x, best_p, float(ys[0])


# ─────────────────────────────────────────────────────
# SHAPE
# ─────────────────────────────────────────────────────

def _foreground(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Binary foreground mask of the marker glyph within its cell.

    Foreground = saturated (colored markers) OR markedly darker/lighter
    than the local grey row background (grey markers: triangle, plus, rings).
    """
    bgr = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2].astype(np.int16)

    colored = s > _SAT_MIN
    # Background grey level = median value of the cell border.
    border = np.concatenate([v[0, :], v[-1, :], v[:, 0], v[:, -1]])
    bg = float(np.median(border))
    grey_fg = np.abs(v - bg) > _FG_GREY_DELTA

    fg = (colored | grey_fg).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return fg


_UPSCALE = _CONFIG.get("glyph_upscale", 6)
_TEMPLATE_SIZE = _CONFIG.get("template_size", 24)


def _glyph_mask(cell_rgb: np.ndarray) -> np.ndarray:
    """Upscaled, centered binary mask of the marker glyph (uint8 0/1)."""
    mask, _, _ = _read_glyph(cell_rgb)
    return mask


def _read_glyph(
    cell_rgb: np.ndarray, z: Optional[int] = None, sat_only: bool = False,
) -> tuple[np.ndarray, Optional[float], np.ndarray]:
    """
    Isolate the central marker glyph and read its color, all in upscaled space.

    Small marker glyphs are upscaled (cubic) so shape is analyzable; the
    upscale factor `z` is chosen so the glyph reaches a workable size at any
    source resolution. We build a foreground mask from saturation and
    grey-contrast, keep the component nearest the cell center (discarding
    gridlines / class-name text at the edges), then read the glyph's hue from
    its own colored pixels.

    Returns:
        (mask, hue, hsv_up) where mask is the upscaled glyph (uint8 0/1),
        hue is the circular-mean hue or None (grey), and hsv_up is the
        upscaled HSV image (for color sampling).
    """
    if z is None:
        side = max(cell_rgb.shape[0], cell_rgb.shape[1], 1)
        z = max(2, int(round(48 / side)))
    big = cv2.resize(
        cell_rgb, (cell_rgb.shape[1] * z, cell_rgb.shape[0] * z),
        interpolation=cv2.INTER_CUBIC,
    )
    bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    border = np.concatenate([v[0, :], v[-1, :], v[:, 0], v[:, -1]])
    bg = float(np.median(border))

    # On the aerial photo the background is not grey, so grey-contrast would
    # pull in background; use saturation alone there. In the dialog (grey
    # cells) include grey-contrast so grey markers are captured too.
    if sat_only:
        distinct = s / 255.0
    else:
        distinct = np.maximum(s / 255.0, np.abs(v - bg) / 255.0)
    thr = max(0.18, float(distinct.max()) * 0.45)
    fg = (distinct >= thr).astype(np.uint8)
    # Close to join fragments (asterisk/plus arms), but do NOT open — opening
    # erodes the thin arms of asterisks/plus/outline markers entirely.
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    mask = _central_component(fg, z)

    # Hue from the glyph's own saturated pixels (in upscaled space). Thin
    # markers have few colored pixels, so the bar is low.
    gm = mask > 0
    colored = gm & (s > _SAT_MIN) & (v > _VAL_MIN)
    if int(colored.sum()) >= max(3, z):
        hue = _circular_mean_hue(hsv[:, :, 0][colored])
    else:
        hue = None
    return mask, hue, hsv


def _central_component(fg: np.ndarray, z: int) -> np.ndarray:
    """Keep the foreground component nearest the cell center."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return fg
    h, w = fg.shape
    cx0, cy0 = w / 2.0, h / 2.0
    best, best_score = None, None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < z * z:
            continue
        cx, cy = cents[i]
        dist = abs(cx - cx0) + abs(cy - cy0)
        score = dist - 0.05 * min(area, 400)
        if best_score is None or score < best_score:
            best, best_score = i, score
    if best is None:
        return np.zeros_like(fg)
    return (labels == best).astype(np.uint8)


def canonical_template(mask: np.ndarray, size: int = _TEMPLATE_SIZE) -> np.ndarray:
    """
    Center a glyph mask on its centroid and scale-normalize to (size, size).

    Produces a translation/scale-invariant float template in [0,1] for
    matching aerial dots against legend markers.
    """
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros((size, size), np.float32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    # Pad to square, then resize, preserving aspect ratio.
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    sq[oy:oy + h, ox:ox + w] = crop
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def _classify_shape(fg: np.ndarray) -> str:
    """Classify a small marker glyph from its foreground mask."""
    ys, xs = np.where(fg > 0)
    if xs.size < 3:
        return "unknown"

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    bbox_area = bw * bh
    fill = xs.size / max(bbox_area, 1)

    sub = fg[y0:y1 + 1, x0:x1 + 1]

    # Hole detection: a background region enclosed by foreground => ring/outline.
    has_hole = _has_enclosed_hole(sub)

    contours, _ = cv2.findContours(
        sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    solidity = 0.0
    n_vertices = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        harea = cv2.contourArea(hull)
        if harea > 0:
            solidity = area / harea
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        n_vertices = len(approx)

    # Circularity (1.0 = perfect circle, ~0.785 = square, lower = spiky).
    circ = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        if peri > 0:
            circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri)

    corner = _corner_fill(sub)
    axis = _axis_fraction(sub)

    # 1. Outline shapes (hollow center): triangle vs ring.
    if has_hole:
        return "triangle" if n_vertices == 3 else "ring"

    # 2. Plus/cross: nearly all pixels lie on the central vertical/horizontal
    #    axes (empty diagonals), corners empty, low fill.
    if fill < 0.6 and corner < 0.35 and axis > 0.78 and _is_cross(sub):
        return "plus"

    # 3. Star/asterisk: radiating arms (incl. diagonals) => low solidity/circ,
    #    empty corners, but pixels are NOT confined to the axes.
    if corner < 0.5 and solidity < 0.7 and circ < 0.62 and axis < 0.82:
        return "star"

    # 4. Filled convex shapes: a square fills its corners, a circle does not.
    if fill >= _FILL_FILLED:
        if corner >= 0.6 and circ < 0.85:
            return "square"
        return "circle"

    # 5. Low-fill fallback.
    return "star" if (solidity < 0.7 and corner < 0.5) else "circle"


def _axis_fraction(sub: np.ndarray) -> float:
    """
    Fraction of foreground lying on the central horizontal/vertical axes.

    A plus concentrates all pixels on the two axes (empty diagonals) => ~1.0.
    An asterisk/star has diagonal arms too, so a meaningful share is off-axis.
    """
    total = int(sub.sum())
    if total == 0:
        return 0.0
    h, w = sub.shape
    band_h = max(1, int(h * 0.2))
    band_w = max(1, int(w * 0.2))
    cy, cx = h // 2, w // 2
    on = np.zeros_like(sub)
    on[max(0, cy - band_h):cy + band_h + 1, :] = sub[max(0, cy - band_h):cy + band_h + 1, :]
    on[:, max(0, cx - band_w):cx + band_w + 1] = sub[:, max(0, cx - band_w):cx + band_w + 1]
    return float(int(on.sum()) / total)


def _corner_fill(sub: np.ndarray) -> float:
    """
    Mean foreground occupancy of the four bbox corners.

    A filled square fills its corners (~1.0); a filled circle leaves them
    empty (~0.1). Used to separate square from circle.
    """
    h, w = sub.shape
    cy, cx = max(1, h // 3), max(1, w // 3)
    corners = [
        sub[:cy, :cx], sub[:cy, w - cx:],
        sub[h - cy:, :cx], sub[h - cy:, w - cx:],
    ]
    return float(np.mean([c.mean() for c in corners if c.size]))


def _has_enclosed_hole(sub: np.ndarray) -> bool:
    """True if the glyph encloses a background hole (ring/triangle outline)."""
    h, w = sub.shape
    if h < 4 or w < 4:
        return False
    inv = (sub == 0).astype(np.uint8)
    # Flood fill background from the border; enclosed bg stays unfilled.
    ff = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if ff[0, x]:
            cv2.floodFill(ff, mask, (x, 0), 0)
        if ff[h - 1, x]:
            cv2.floodFill(ff, mask, (x, h - 1), 0)
    for y in range(h):
        if ff[y, 0]:
            cv2.floodFill(ff, mask, (0, y), 0)
        if ff[y, w - 1]:
            cv2.floodFill(ff, mask, (w - 1, y), 0)
    return int(ff.sum()) >= 1


def _is_marker_like(glyph: np.ndarray, aerial: bool = False) -> bool:
    """
    True if the isolated glyph mask is a plausible marker.

    `glyph` is the upscaled central-component mask from `_glyph_mask`. We
    require a compact blob whose area is a sane fraction of the cell and that
    does not span the entire cell (border bleed).

    `aerial`: the `frac > 0.85` and full-cell-span guards exist to reject
    border-bleed in a *dialog cell*, where a real marker sits inside a grey
    cell with margin. An aerial dot has no surrounding cell, so it legitimately
    fills its own tight crop — those two guards then reject the majority of real
    dots (measured: A 31% pass, C 49% pass at default crop). With aerial=True
    they are relaxed: only near-total fill (>0.97, i.e. no shape boundary at
    all) and the noise / off-centre guards remain.
    """
    ys, xs = np.where(glyph > 0)
    if xs.size == 0:
        return False
    h, w = glyph.shape
    frac = xs.size / float(h * w)
    hi = 0.97 if aerial else 0.85
    if frac < 0.02 or frac > hi:
        return False
    if not aerial:
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        if bw >= w and bh >= h:
            return False
    # Centroid near cell center.
    if abs(xs.mean() - w / 2) > w * 0.42 or abs(ys.mean() - h / 2) > h * 0.42:
        return False
    return True


def _is_cross(sub: np.ndarray) -> bool:
    """Detect a plus/cross: a dominant central row and column of foreground."""
    h, w = sub.shape
    col_sum = sub.sum(axis=0)
    row_sum = sub.sum(axis=1)
    cx = int(np.argmax(col_sum))
    cy = int(np.argmax(row_sum))
    # Center bias + both arms present.
    central = (0.2 * w < cx < 0.8 * w) and (0.2 * h < cy < 0.8 * h)
    strong_v = col_sum[cx] >= 0.6 * h
    strong_h = row_sum[cy] >= 0.6 * w
    return central and strong_v and strong_h


# ─────────────────────────────────────────────────────
# DIALOG LOCALIZATION (floating window, variable position)
# ─────────────────────────────────────────────────────

def _all_colored_blobs(rgb: np.ndarray) -> list[tuple[float, float]]:
    """Centroids of all small uniform colored blobs in the full screenshot."""
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > _BLOB_SAT_MIN) & (v > _VAL_MIN)).astype(np.uint8) * 255
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 6 or a > 600:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > 3:
            continue
        out.append((float(cents[i][0]), float(cents[i][1])))
    return out


def _local_std(gray: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-pixel local standard deviation (texture measure)."""
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, -1, (k, k))
    sq = cv2.boxFilter(g * g, -1, (k, k))
    return np.sqrt(np.maximum(sq - m * m, 0))


def _all_colored_sized(rgb: np.ndarray) -> list[tuple[float, float, float]]:
    """Colored blobs as (cx, cy, size) over the whole screenshot."""
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > _BLOB_SAT_MIN) & (v > _VAL_MIN)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 8 or a > 700:
            continue
        if max(bw, bh) / max(min(bw, bh), 1) > 3:
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), float(max(bw, bh))))
    return out


def _column_ladder(
    inside: list[tuple[float, float, float]], msize: float,
) -> tuple[int, Optional[float], float, float, float]:
    """
    Best tight-x-column regular ladder among blobs.

    Returns (inliers, marker_x, y_first, y_last, pitch). Dialog markers form a
    tight column with a regular pitch; scattered aerial dots do not.
    """
    if len(inside) < _LADDER_MIN:
        return 0, None, 0.0, 0.0, 0.0
    xs = np.array([b[0] for b in inside])
    best = (0, None, 0.0, 0.0, 0.0)
    for seed in np.unique(np.round(xs)):
        col = [b for b in inside if abs(b[0] - seed) <= max(6, msize * 0.7)]
        if len(col) < _LADDER_MIN:
            continue
        ys = np.sort(np.array([b[1] for b in col]))
        diffs = np.diff(ys)
        diffs = diffs[(diffs >= 6) & (diffs <= 60)]
        if diffs.size < 2:
            continue
        pitch = float(np.median(diffs))
        k = np.round((ys - ys[0]) / pitch)
        pred = ys[0] + k * pitch
        inl = int((np.abs(ys - pred) <= pitch * 0.3).sum())
        if inl > best[0]:
            best = (inl, float(np.median([b[0] for b in col])),
                    float(ys[0]), float(ys[-1]), pitch)
    return best


def locate_dialog(rgb: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Locate the floating 'Manual Point Count' dialog in a full screenshot.

    The dialog can sit anywhere (top-right, right, bottom-right). We find the
    flat-grey UI panels (the window body is flat; sand/vegetation are textured)
    that are a contained box AND contain a tight, regularly-pitched column of
    colored markers (the legend). The bbox is then built from that marker
    column + its row span, so the rest of the image stays available as aerial —
    this avoids the old decomposer's vertical split that discarded ~half the
    aerial (and its annotations).

    Returns (x, y, w, h) bounding the dialog, or None if not found.
    """
    h, w = rgb.shape[:2]
    img_area = h * w
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    v = rgb[:, :, 0]
    gray = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    grey = (np.abs(r - g) < 18) & (np.abs(g - b) < 18) & (v > 185) & (v < 236)
    flat = _local_std(gray, 5) < 8
    ui = (grey & flat).astype(np.uint8)
    ui = cv2.morphologyEx(
        ui, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    ui = cv2.morphologyEx(ui, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(ui, 8)
    blobs = _all_colored_sized(rgb)
    if len(blobs) < 6:
        return None
    msize = float(np.median([z[2] for z in blobs]))

    best = None  # (inliers, marker_x, y0, y1, pitch, panel_right)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        if a < 0.01 * img_area or a > 0.40 * img_area:
            continue  # a dialog is a contained box, not the whole frame
        if bw < 130 or bh < 90:
            continue
        inside = [z for z in blobs if x <= z[0] <= x + bw and y <= z[1] <= y + bh]
        inl, mx, y0, y1, pitch = _column_ladder(inside, msize)
        if inl >= _LADDER_MIN and (best is None or inl > best[0]):
            best = (inl, mx, y0, y1, pitch, x + bw)

    if best is None or best[1] is None:
        return None
    inl, mx, y0, y1, pitch, panel_right = best
    x0 = int(max(0, mx - msize * 11))
    # Right edge: the marker-relative offset (mx + msize*30) falls short when the
    # Name column is wide -- long class names push the Count column further right
    # (e.g. image B: 'BRPE chick nest w/o adult'), so a fixed offset cropped the
    # entire Count column and made the counts look blank. Extend to the grey
    # panel's own right edge when that is wider; the panel is flat grey, so this
    # never reaches into the (textured) aerial.
    x1 = int(min(w, max(mx + msize * 30, panel_right)))
    yy0 = int(max(0, y0 - pitch * 3.5))
    yy1 = int(min(h, y1 + pitch * 2.5))
    box = (x0, yy0, x1 - x0, yy1 - yy0)

    # "A dialog is a contained box, not the whole frame" is checked above on the grey
    # COMPONENT, but the box returned is rebuilt from the marker column and can grow
    # far past it. On 17May15Camera1-Card3-01497 a column of aerial markers formed a
    # ladder and the emitted box covered 54% of the frame -- almost all photograph,
    # with a sliver of the real dialog at its right edge -- from which parse_legend
    # read 25 phantom rows out of vegetation. Applying the bound the function already
    # declares, to the thing it actually returns, rejects that and the same failure on
    # 00948, 02092, 00622 and 1216.
    if box[2] * box[3] > _DIALOG_MAX_AREA * img_area:
        logger.info("legend: rejecting a %dx%d box covering %.0f%% of the frame",
                    box[2], box[3], 100 * box[2] * box[3] / img_area)
        return None
    return box


def parse_screenshot(rgb: np.ndarray) -> tuple[list[LegendEntry], Optional[tuple]]:
    """
    Locate the dialog in a full screenshot and parse its legend.

    Returns (entries, dialog_bbox). Empty entries + None bbox if no dialog
    could be located.
    """
    bbox = locate_dialog(rgb)
    if bbox is None:
        return [], None
    x, y, bw, bh = bbox
    crop = rgb[y:y + bh, x:x + bw]
    return parse_legend(crop), bbox


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def parse_legend(
    dialog_rgb: np.ndarray, cell: Optional[int] = None
) -> list[LegendEntry]:
    """
    Parse the legend table from a dialog region.

    Args:
        dialog_rgb: Dialog region (H, W, 3) uint8 RGB (from decompose).
        cell: Half-size used to crop each marker glyph. If None, derived from
              the detected row pitch (scale-adaptive across resolutions).

    Returns:
        Ordered list of LegendEntry (top to bottom). Empty if no grid found.
    """
    if dialog_rgb is None or dialog_rgb.size == 0:
        return []

    dh, dw = dialog_rgb.shape[:2]
    bgr = cv2.cvtColor(dialog_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blobs, msize = _marker_blobs(hsv)
    marker_x, pitch, ref_y = _fit_grid(blobs, msize, dh)
    if marker_x is None or pitch is None or ref_y is None:
        logger.warning("legend: could not fit row grid (blobs=%d)", len(blobs))
        return []

    # Crop half-size: a touch over half the row pitch, so each cell captures
    # one marker without bleeding into neighbouring rows.
    if cell is None:
        cell = max(4, int(round(pitch * 0.5)))
    band_pad = max(_MARKER_BAND_PAD, msize * 0.8)

    # Colored marker anchors in the marker column, sorted top to bottom.
    band = sorted(
        ((b[0], b[1]) for b in blobs if abs(b[1] - marker_x) <= band_pad),
        key=lambda t: t[0],
    )
    origin_y = band[0][0]  # topmost real marker = first table row
    mx = int(round(marker_x))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    text_w = max(40, int(pitch * 6))

    def row_has_name_text(y: int) -> bool:
        """
        Class-name text exists to the right of the marker column.

        This decides how far past the last coloured marker the table continues, so
        anything it accepts becomes a legend row. Counting dark pixels was far too
        weak: below the table on `17May10Camera2-Card1-5745` it accepted the blank
        gap and the horizontal scrollbar, inventing two rows the dialog does not
        have. Thirty-one aerial dots were then assigned to a class that does not
        exist — the single largest remaining classification error.

        Text is several separate marks. Measured on that dialog's name column, real
        rows put ink in 41-54 distinct columns spread over many runs; the blank gap
        manages 1-5; the scrollbar covers 70 columns in one unbroken run. So require
        both some width of ink and that it is broken up, which a rule or a scrollbar
        never is.
        """
        x_lo = min(dw - 1, mx + cell + 2)
        x_hi = min(dw, mx + cell + text_w)
        if x_hi - x_lo < 6:
            return False
        strip = gray[max(0, y - int(pitch * 0.3)):y + int(pitch * 0.3) + 1, x_lo:x_hi]
        if int((strip < 95).sum()) < 4:
            return False
        if _TEXT_MIN_COLS <= 0:
            return True
        # Text is several separate marks. A blank gap puts ink in a handful of
        # columns and a scrollbar puts it in one unbroken run, so require both a
        # width of ink and that it is broken up.
        cols = np.where((strip < 95).any(axis=0))[0]
        if cols.size < _TEXT_MIN_COLS:
            return False
        return int(np.count_nonzero(np.diff(cols) > 1)) + 1 >= _TEXT_MIN_RUNS

    # Row index of each colored blob, relative to the topmost marker.
    # Blobs anchor their rows precisely; the grid only numbers rows and
    # fills grey-marker gaps, so the alignment cannot slip.
    blob_by_row: dict[int, tuple[float, float]] = {}
    for by, bx in band:
        r = int(round((by - origin_y) / pitch))
        # Keep the blob closest to the ideal row center if two collide.
        if r not in blob_by_row or abs(by - (origin_y + r * pitch)) < abs(
            blob_by_row[r][0] - (origin_y + r * pitch)
        ):
            blob_by_row[r] = (by, bx)

    # Determine the table's last data row by walking the class-name text
    # column downward until two consecutive rows have no text.
    last_row = max(blob_by_row)
    gap = 0
    r = last_row
    while True:
        r += 1
        yc = origin_y + r * pitch
        if yc > dh - cell:
            break
        if row_has_name_text(int(round(yc))):
            last_row = r
            gap = 0
        else:
            gap += 1
            if gap >= 2:
                break

    half = cell
    entries: list[LegendEntry] = []

    for r in range(0, last_row + 1):
        if r in blob_by_row:
            cyc, cxc = blob_by_row[r]
        else:
            cyc, cxc = origin_y + r * pitch, marker_x
        y = int(round(cyc))
        cxi = int(round(cxc))
        if not (cell <= y <= dh - cell):
            continue

        crop = dialog_rgb[
            max(0, y - half):y + half + 1, max(0, cxi - half):cxi + half + 1
        ]
        if crop.size == 0:
            continue

        glyph, hue, _ = _read_glyph(crop)
        if not _is_marker_like(glyph):
            # Grey/low-contrast marker the glyph extractor missed: still emit
            # the row (anchored by the grid) so class ordering stays intact.
            shape, color = "unknown", "grey"
            template = canonical_template(glyph)
        else:
            shape = _classify_shape(glyph)
            color = _name_hue(hue) if hue is not None else "grey"
            template = canonical_template(glyph)

        entries.append(
            LegendEntry(
                row=r,
                cy=float(y),
                cx=float(cxi),
                shape=shape,
                color=color,
                hue=None if hue is None else round(hue, 1),
                marker=crop.copy(),
                template=template,
            )
        )

    logger.info(
        "legend: parsed %d rows (pitch=%.1f, x=%d, last_row=%d)",
        len(entries), pitch, mx, last_row,
    )
    return entries


# ─────────────────────────────────────────────────────
# CLASS-NAME OCR (attach the text label to each marker)
# ─────────────────────────────────────────────────────

# Known category words (the suffix after the species code). Lower-cased.
_CATEGORIES = [
    "wbn", "site", "bird", "chick", "nest", "adult", "ad", "stand", "roost",
    "brood", "imm", "pbn", "abandn", "aband", "empty", "nestling", "sit",
    "float", "floating", "colony", "nestw", "chickn-ad", "chickn", "duck",
    "stand", "ad roost", "imm roost", "in colony",
]


def _ocr_engine():
    """Return a configured pytesseract module, or None if unavailable."""
    try:
        import pytesseract
    except ImportError:
        return None
    # Locate the tesseract binary on Windows if not already on PATH.
    import shutil
    if shutil.which("tesseract") is None:
        for cand in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if Path(cand).exists():
                pytesseract.pytesseract.tesseract_cmd = cand
                break
        else:
            return None
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None
    return pytesseract


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two short strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _best_match(token: str, vocab: list[str], max_dist: int) -> Optional[str]:
    """Closest vocabulary entry to `token` within max_dist edits (case-insensitive)."""
    if not token:
        return None
    t = token.lower()
    best, best_d = None, max_dist + 1
    for w in vocab:
        d = _levenshtein(t, w.lower())
        if d < best_d:
            best_d, best = d, w
    return best if best_d <= max_dist else None


def _find_species(tokens: list[str], vocab: list[str]) -> tuple[
    Optional[str], Optional[int], str
]:
    """
    Locate a species code in the leading tokens, tolerant of OCR punctuation and
    species+category gluing. Small-font OCR routinely glues the 4-letter code to
    punctuation or to the next word ('__LAGU' -> LAGU, 'NECO-AD' -> NECO,
    'BRPEWBN' -> BRPE), which inflates the edit distance past max_dist=1 so the
    raw first token never matches.

    We strip non-alpha noise from each of the first two tokens and try the whole
    cleaned token, then its 4-letter prefix (every species code is 4 letters).

    The length gate is 3, not 4, because OCR routinely drops the leading character
    of a small-font code and a token one short then never reaches the matcher at
    all. Measured over the 25 frames that reach classification, that alone accounts
    for five rows — `RPE` and `BRE` are both one edit from `BRPE`.

    It stays at edit distance 1. Raising it to 2 would resolve two more rows and be
    wrong on both: `nest` is a category word and sits two edits from the species
    code `BNST`, so a looser matcher invents a species that is not there. Coverage
    bought with a wrong species is a regression, not a gain.

    Returns (species, token_index, glued_remainder) where glued_remainder is the
    leftover category text that shared the species token (e.g. 'WBN' from
    'BRPEWBN'), or ''.
    """
    for idx in range(min(2, len(tokens))):
        a = re.sub(r"[^A-Za-z]", "", tokens[idx])
        if len(a) < _SPECIES_MIN_TOKEN:
            continue
        m = _best_match(a, vocab, max_dist=1)
        if m:
            return m, idx, ""
        m = _best_match(a[:4], vocab, max_dist=1)
        if m:
            return m, idx, a[4:]
    return None, None, ""


def _parse_class_text(text: str, species_vocab: list[str]) -> tuple[
    Optional[str], Optional[str], Optional[str], Optional[int]
]:
    """
    Parse one OCR'd legend row into (class_name, species, category, count).

    Fuzzy-matches the leading token to a species code and the remainder to a
    category word, correcting common small-font OCR errors (BAPE->BRPE).
    """
    raw = " ".join(text.split())
    if not raw:
        return None, None, None, None

    tokens = raw.replace("|", "").split()
    count = None
    # Trailing integer = the Count column value.
    if tokens and tokens[-1].lstrip("-").isdigit():
        count = int(tokens[-1])
        tokens = tokens[:-1]
    if not tokens:
        return None, None, None, count

    species, sidx, glued = _find_species(tokens, species_vocab)
    if species is not None:
        rest = tokens[sidx + 1:]
        if glued:
            rest = [glued] + rest
    else:
        rest = tokens
    category = None
    if rest:
        joined = " ".join(rest)
        category = (
            _best_match(rest[0], _CATEGORIES, max_dist=2)
            or _best_match(joined, _CATEGORIES, max_dist=3)
            or joined
        )
    name_bits = [b for b in (species, category) if b]
    class_name = " ".join(name_bits) if name_bits else (raw or None)
    return class_name, species, category, count


def _column_gridlines(gray: np.ndarray) -> list[int]:
    """
    X-positions of the table's vertical gridlines (full-height dark columns).

    The legend table is marker | Name | Count separated by thin vertical
    borders that are dark over (nearly) the full dialog height, unlike text.
    Used to split the Name strip from the Count strip robustly at any width --
    the old marker-relative offset (cx + pitch*8) fails when long class names
    widen the Name column (image B), which cropped the Count column entirely.
    Nearby columns are merged (double borders / scrollbar edges).
    """
    frac = (gray < 130).mean(axis=0)
    cols = list(np.where(frac >= 0.7)[0])
    groups: list[list[int]] = []
    for x in cols:
        if groups and x - groups[-1][-1] <= 6:
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return [int(np.mean(g)) for g in groups]


def attach_class_names(
    dialog_rgb: np.ndarray,
    entries: list[LegendEntry],
    species_vocab: Optional[list[str]] = None,
) -> list[LegendEntry]:
    """
    Read each legend row's class-name text via OCR and fill it into the entries.

    Self-contained (no CSV): the class label sits to the right of the marker on
    the same row. We OCR each row strip, then fuzzy-match the species code and
    category against known vocabularies to correct small-font OCR errors.

    Mutates and returns `entries`. If OCR is unavailable, entries are unchanged.
    """
    ocr = _ocr_engine()
    if ocr is None:
        logger.warning("attach_class_names: tesseract unavailable, skipping OCR")
        return entries
    if not entries:
        return entries
    if species_vocab is None:
        species_vocab = _SPECIES_CODES

    dh, dw = dialog_rgb.shape[:2]
    gray = cv2.cvtColor(cv2.cvtColor(dialog_rgb, cv2.COLOR_RGB2BGR),
                        cv2.COLOR_BGR2GRAY)

    cys = sorted(e.cy for e in entries)
    pitch = float(np.median(np.diff(cys))) if len(cys) > 1 else 14.0
    half = max(4, int(pitch * 0.45))

    # Split Name from Count at the table's own gridlines. Columns run
    # [table-left | Name(+marker) | Count | (scrollbar)], so left-to-right the
    # gridlines are: [0]=table left, [1]=Name|Count divider, [2]=Count right
    # border, [3]=scrollbar (B only). Counting from the LEFT is robust to the
    # scrollbar; counting from the right mis-picks the empty margin on B. Adapts
    # to any Name-column width (fixes image B, where long names pushed the Count
    # column past the old marker-relative offset).
    #
    # ## Known wrong on 14 of the 25 frames, and TWICE reverted — read before retrying
    #
    # This indexing assumes the first gridline is the table's left border and that
    # at least three are found. `5745` yields only `[77, 174]`, so the divider is
    # dropped and the name strip runs the full width, pulling Count digits into the
    # name OCR. `0449` yields `[8, 110, 195, 315]` against a marker at x=122, so
    # index 1 lands to the *left* of the marker: what the code reads as the Count
    # column is actually the Name column, and a dialog plainly showing 93/70/11/23/10
    # comes back as 3/None/0/0/0. Capacity caps at those, leaving 66 of 83 labelled
    # dots unassigned — that frame scores 0.193 against 0.78 pooled.
    #
    # Choosing the first gridline right of the marker is clearly the correct
    # geometry, and it improves names both times it was tried (species 0.771 →
    # 0.858). It has broken classification both times:
    #
    #   moving both strips together   count 0.693 → 0.624, `5745` 0.861 → 0.634
    #   bounding the strips apart     count 0.693 → 0.683, `5745` 0.861 → 0.600
    #
    # The second attempt kept the Count column on its old marker-relative offset
    # unless gridlines bounded it on both sides, and `5745` still fell — so the
    # damage is not only the Count strip. Moving the NAME boundary changes what
    # `_parse_class_text` sees, which changes `class_name`, which changes the row
    # identities the matcher and the eval both key on.
    #
    # So: do not retry this as a legend-parsing change scored on name coverage. It
    # is a classification change and must be scored on `eval_localisation.py`, with
    # `5745` (0.861) and `0027` (0.877) as the gate.
    gridlines = _column_gridlines(gray)
    if len(gridlines) >= 3 and gridlines[2] - gridlines[1] >= 15:
        name_count_div, right_border = gridlines[1], gridlines[2]
    else:
        name_count_div, right_border = None, dw

    for e in entries:
        y0 = max(0, int(e.cy) - half)
        y1 = min(dh, int(e.cy) + half + 1)
        x0 = min(dw - 1, int(e.cx) + int(pitch * 0.7))
        # Bound the name strip at the Name|Count divider so count digits do not
        # corrupt the name OCR (and vice-versa).
        name_r = name_count_div if (name_count_div and name_count_div > x0 + 5) else dw
        strip = gray[y0:y1, x0:name_r]
        if strip.size == 0 or strip.shape[1] < 8:
            continue
        text = _ocr_line(ocr, strip)
        name, sp, cat, cnt = _parse_class_text(text, species_vocab)
        # Dedicated digit-only pass on the Count column (digit whitelist). Use
        # the detected column when available, else fall back to the marker offset.
        if name_count_div is not None:
            cstrip = gray[y0:y1, name_count_div + 2:right_border]
        else:
            cx0 = min(dw - 1, int(e.cx) + int(pitch * 8))
            cstrip = gray[y0:y1, cx0:dw]
        cnt2 = _ocr_count(ocr, cstrip)
        e.class_name, e.species, e.category = name, sp, cat
        e.count = cnt2 if cnt2 is not None else cnt

    return entries


def _ocr_line(ocr, strip: np.ndarray) -> str:
    """OCR a single row strip of small UI text (upscale + binarize + pad)."""
    big = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # A quiet border helps Tesseract segment edge characters.
    big = cv2.copyMakeBorder(big, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    try:
        return ocr.image_to_string(big, config="--psm 7")
    except Exception:
        return ""


def _ocr_count(ocr, strip: np.ndarray) -> Optional[int]:
    """
    Read the integer in a Count-column strip (digit whitelist).

    For wide strips (> 130 px) the digit is right-justified far from the left
    edge — the left portion may be class-name text that corrupts OCR even with
    a digit whitelist.  Clipping to the rightmost 80 px aligns the window with
    the actual digit.  Narrower strips are left unchanged (right position is
    already calibrated by pitch*8).

    A blank read is treated as 0 only when the strip has very few dark pixels
    (truly empty count cells).
    """
    if strip.size == 0 or strip.shape[1] < 6:
        return None
    # Wide strips: count is right-justified; the left portion is class-name.
    # Threshold at 145 px: B-style dialogs (185 px) clip, A/C/D (≤ 141 px) do not.
    if strip.shape[1] > 145:
        strip = strip[:, -80:]

    big = cv2.resize(strip, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    big = cv2.copyMakeBorder(big, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        txt = ocr.image_to_string(big, config=cfg)
    except Exception:
        return None
    digits = "".join(ch for ch in txt if ch.isdigit())
    if digits:
        return int(digits)
    # Blank read: 0 if the cell really is near-empty, else unknown.
    dark = int((strip < 110).sum())
    return 0 if dark < 6 else None


# Species codes (from the dataset). Used to correct OCR of the species token.
_SPECIES_CODES = [
    "AMAV", "AMCO", "AMOY", "ANHI", "AWPE", "BBPL", "BBWD", "BCNH", "BLSK",
    "BLTE", "BNST", "BRNO", "BRPE", "BWTE", "CAEG", "CANG", "CARO", "CATE",
    "COGA", "COTE", "CRCA", "DAIB", "DCCO", "DUCK", "FICR", "FOCO", "FOTE",
    "FUWD", "GBHE", "GBTE", "GLIB", "GREG", "GRFL", "GRHE", "GTGR", "HERG",
    "LAGU", "LBBG", "LBHE", "LEBI", "LETE", "MABO", "MAFR", "MAGO", "MODU",
    "NECO", "NSHO", "OSPR", "RBGU", "REEG", "REKN", "ROSA", "ROSP", "ROST",
    "ROYT", "RSHA", "RUTU", "SATE", "SBDO", "SDHE", "SNEG", "SONO", "SOTE",
    "TRHE", "TRSN", "ULGU", "ULTE", "UNCO", "UNCR", "UNDU", "UNEG", "UNGT",
    "UNGU", "UNHG", "UNIB", "UNID", "UNNH", "UNRA", "UNSB", "UNSH", "UNTE",
    "UNWA", "UNWW", "USTE", "WADE", "WFIB", "WHEG", "WHIB", "WILL", "WIPH",
    "WIPL", "WOST", "YCNH",
]
