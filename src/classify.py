"""
Classify aerial annotation dots against a per-image legend.

Stage that turns the parsed legend (src/legend.py) into separated, labelled
detections: each colored dot baked into the aerial photo is matched to one of
the legend markers by COLOUR first, then — when several classes share a colour
(e.g. red circle "BRPE WBN" vs red plus "BRPE bird") — by SHAPE via template
correlation against the legend glyph. This keeps categories that share a colour
distinct, which colour-only detection cannot do.

Input: aerial RGB + parsed legend entries (with templates).
Output: list[AerialDot], each assigned to a legend class.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter

if TYPE_CHECKING:  # avoid a runtime import cycle (subtract imports classify)
    from src.align import AlignResult

from src.legend import (
    LegendEntry,
    _read_glyph,
    _central_component,
    _classify_shape,
    _is_marker_like,
    _name_hue,
    _circular_mean_hue,
    _SAT_MIN,
    _VAL_MIN,
    canonical_template,
)

logger = logging.getLogger(__name__)

# ── Per-image colour anchoring ──────────────────────────────────────────────
# Josh's ask: assign each aerial dot to the dialog's OWN palette, not fixed
# global hue bins. Step-0 measurement (scripts/diagnose_marker_colors.py) showed
# hue barely varies within a colour group; the class-separating signal (e.g.
# light vs dark red) lives in BRIGHTNESS. So the anchor space must include value.
#
# For each aerial dot we compute a colour vector, then take as candidates the
# legend entries whose colour is within _COLOR_MARGIN of the closest one (a
# self-adapting cluster: same-colour classes stay together, a far light/dark
# variant drops out). Shape/template then separates same-colour candidates.
# Dots whose closest legend colour is beyond _COLOR_REJECT are left unassigned
# (off-palette noise) instead of being forced onto a wrong class.
_COLOR_SPACE = "lab"     # "lab" (perceptual, uniform threshold) or "hsv"
_COLOR_MARGIN = 22.0     # candidates within (min_dist + margin) of the dot
_COLOR_REJECT = 45.0     # closest colour beyond this -> unassigned
_UNSET = object()        # cache sentinel (avoids array-truthiness on None/ndarray)


@dataclass
class AerialDot:
    """A detected annotation dot in the aerial, assigned to a legend class."""
    cx: float
    cy: float
    color: str
    shape: str
    area: int
    template: np.ndarray
    quality: float = 0.0          # how dot-like (saturation/contrast); for ranking
    color_vec: Optional[np.ndarray] = None   # colour signature for palette anchoring
    # Filled in by assignment:
    legend_row: Optional[int] = None   # LegendEntry.row — the class IDENTITY
    species: Optional[str] = None
    category: Optional[str] = None
    class_name: Optional[str] = None   # display text; may be None or non-unique
    match_score: float = 0.0


def _split_cluster(
    blob_mask: np.ndarray, n_expected: int,
) -> list[tuple[float, float]]:
    """
    Split a merged cluster of overlapping dots into individual centers.

    Overlapping annotation dots form one connected component. A distance
    transform peaks at each dot center; local maxima recover the individuals.
    Returns a list of (cx, cy) in blob_mask coordinates.
    """
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    # Peak-separation scaled to the expected per-dot footprint.
    fs = max(3, int(np.sqrt(blob_mask.sum() / max(n_expected, 1)) * 0.9))
    if fs % 2 == 0:
        fs += 1
    peaks = (dist == maximum_filter(dist, size=fs)) & (dist > 2.0)
    labeled, n_found = scipy_label(peaks)
    if n_found == 0:
        ys, xs = np.where(blob_mask > 0)
        return [(float(xs.mean()), float(ys.mean()))] if xs.size else []
    centers = []
    for j in range(1, n_found + 1):
        ys, xs = np.where(labeled == j)
        if xs.size:
            centers.append((float(xs.mean()), float(ys.mean())))
    return centers


def _dot_centers(
    rgb: np.ndarray, exclude: Optional[tuple[int, int, int, int]] = None,
    sat_min: int = 80, val_min: int = 55,
) -> tuple[list[tuple[float, float]], float]:
    """
    Centers of all annotation dots in the aerial, splitting merged clusters.

    Returns (centers, single_area). `exclude` is the dialog (x, y, w, h) box,
    whose dots are dropped so legend markers are not counted.
    """
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s > sat_min) & (v > val_min)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    areas = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 8:
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        if exclude is not None:
            ex, ey, ew, eh = exclude
            if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
                continue
        comps.append((i, area, bw, bh, cx, cy,
                      int(stats[i, cv2.CC_STAT_LEFT]),
                      int(stats[i, cv2.CC_STAT_TOP])))
        if max(bw, bh) / max(min(bw, bh), 1) <= 2:
            areas.append(area)
    if not comps:
        return [], 0.0

    # Single-dot area = median of compact (non-cluster) components.
    single_area = float(np.median(areas)) if areas else float(
        np.median([c[1] for c in comps]))
    single_area = max(single_area, 12.0)

    centers: list[tuple[float, float]] = []
    for (i, area, bw, bh, cx, cy, x0, y0) in comps:
        # Only treat as a cluster when clearly larger than one dot.
        if area < single_area * 1.5:
            if max(bw, bh) / max(min(bw, bh), 1) <= 4:
                centers.append((cx, cy))
            continue
        n_est = int(round(area / single_area))
        sub = (labels[y0:y0 + bh, x0:x0 + bw] == i).astype(np.uint8)
        split = _split_cluster(sub, n_est)
        # Fall back to the single centroid if splitting found nothing useful.
        if split:
            for (lx, ly) in split:
                centers.append((x0 + lx, y0 + ly))
        else:
            centers.append((cx, cy))

    # Re-apply exclude to ALL centers: cluster-split sub-dots near the dialog
    # edge can land inside the box even when their parent component's centroid
    # was outside it (the per-component check above misses those).
    if exclude is not None:
        ex, ey, ew, eh = exclude
        centers = [(cx, cy) for (cx, cy) in centers
                   if not (ex <= cx <= ex + ew and ey <= cy <= ey + eh)]
    return centers, single_area


def _template_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two canonical glyph templates (shape match)."""
    av, bv = a.ravel(), b.ravel()
    na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(av @ bv / (na * nb))


_HUE_TOL = 14   # deg (OpenCV 0-180) around the dot's own hue kept as marker
# Background removal on by default; set BG_REMOVAL=0 to A/B against Stage 1.
_BG_REMOVAL = os.environ.get("BG_REMOVAL", "1") != "0"
# Shape matching: intensity-template NCC, no shape-name boost — the config the
# self-recovery ablation selected (scripts/eval_matching.py). The coarse shape
# LABEL (star often collapses to "plus"/"circle" at aerial scale) reinforced
# wrong classes, so the boost is off; the template carries the signal. Toggles
# kept for reproducing the ablation:
#   SHAPE_MATCH=cosine   -> old binary-mask cosine
#   SHAPE_BOOST=1        -> re-enable the +0.35 shape-name agreement boost
_SHAPE_MATCH = os.environ.get("SHAPE_MATCH", "ncc")   # "ncc" | "cosine"
_SHAPE_BOOST = 0.35 if os.environ.get("SHAPE_BOOST", "0") != "0" else 0.0
# Colour anchoring on by default; COLOR_ANCHOR=0 reverts to global colour-name
# grouping (the pre-rework baseline) for real-aerial before/after measurement.
_COLOR_ANCHOR = os.environ.get("COLOR_ANCHOR", "1") != "0"
# Treat each legend row's Count as a capacity during assignment. RESPECT_COUNTS=0
# reverts to per-dot argmax for A/B measurement. See assign_classes.
_RESPECT_COUNTS = os.environ.get("RESPECT_COUNTS", "1") != "0"
# Per-frame lightness correction (see _lightness_offset). L_OFFSET=0 disables it.
_L_OFFSET = os.environ.get("L_OFFSET", "1") != "0"
_CHROMA_MATCH = 25.0      # a,b distance within which a dot's colour is "this row's"
_OFFSET_MIN_DOTS = 8      # too few matches to trust a median -> no correction
# Second assignment pass using a per-row colour offset. The drift from legend glyph
# to aerial marker is per row, not per frame — see `_row_offsets`. ROW_OFFSET=0
# reverts to the single frame-wide shift for A/B.
_ROW_OFFSET = os.environ.get("ROW_OFFSET", "1") != "0"
# Weight on colour agreement inside the pair score. Colour has always gated
# candidacy; this lets it rank too. 0 disables. See `_pair_score`.
#
# Swept over the four selected, hand-labelled frames (491 dots), pooled:
#   0.00  0.789    0.15  0.816    0.20  0.828    0.25  0.820    0.30  0.822
#   0.60  0.803    1.00  0.789
# 0.15-0.30 is one broad stable band, so the value is not balanced on a peak. 0.20
# is taken from inside it because it is the only point where **no frame falls**:
# 5745 +15 dots, 00620 +4, 06389 and 00825 unchanged. This is a constant fitted on
# the regression frames — see the plan's note on that.
_SCORE_COLOR = float(os.environ.get("SCORE_COLOR", "0.2"))
# Retry a dot whose every candidate row filled up, against any row still inside
# `_COLOR_REJECT`. Only blocked dots are reconsidered. BLOCKED_RETRY=0 disables.
_BLOCKED_RETRY = os.environ.get("BLOCKED_RETRY", "1") != "0"
# Discard a row count larger than the whole frame's detections — it cannot be true.
# COUNT_SANITY=0 disables. See where `capacity` is built in `assign_classes`.
_COUNT_SANITY = os.environ.get("COUNT_SANITY", "1") != "0"
# Give a dot whose colour picks one row a flat 1.0, ahead of everything else in
# the capacity queue. Off: those dots are less likely to be real (69% vs 80%).
_SOLE_WINS = os.environ.get("SOLE_CANDIDATE_WINS", "0") != "0"
# What capacity a row gets when the dialog's Count column never read for it.
#   "tail"  only rows parsed BELOW the last counted row hold nothing  (default)
#   "open"  every such row is unlimited (the original behaviour, kept for A/B)
#   "zero"  every such row holds nothing
#   "share" the unread rows share what the dialog's own total leaves over
# Measured over the four selected, hand-labelled frames (488 dots):
#   open 0.725, zero 0.766, tail 0.766 — and tail keeps 0401 at 0.516 where zero
# drops it to 0.065. See `_uncounted_capacity`.
_UNCOUNTED = os.environ.get("UNCOUNTED_CAP", "tail")


def _color_masked_glyph(
    glyph: np.ndarray, hsv_up: np.ndarray, hue: Optional[float],
) -> np.ndarray:
    """
    Background removal: restrict the glyph to pixels of its OWN colour.

    Josh's ask — "extract the glyph colour and mask using that, otherwise it's
    going to try and match grey." Saturation-only extraction still admits
    off-colour background (grey halo, nearby vegetation) that bloats a small
    marker into a blob and drowns thin arms. We keep only pixels within
    _HUE_TOL of the dot's dominant hue, so the template encodes the marker's
    shape, not the patch. Grey markers (hue is None) are returned unchanged.
    Falls back to the original glyph if masking would erase the marker.
    """
    if hue is None or not _BG_REMOVAL:
        return glyph
    H = hsv_up[:, :, 0].astype(np.int16)
    S = hsv_up[:, :, 1]
    V = hsv_up[:, :, 2]
    dh = np.abs(H - int(round(hue)))
    dh = np.minimum(dh, 180 - dh)
    consistent = (dh <= _HUE_TOL) & (S > _SAT_MIN) & (V > _VAL_MIN)
    m = ((glyph > 0) & consistent).astype(np.uint8)
    if int(m.sum()) < 3:
        return glyph
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _central_component(m, 1)


def _intensity_template(mask: np.ndarray, hsv_up: np.ndarray) -> np.ndarray:
    """
    Canonical template weighted by marker INTENSITY, not a hard binary mask.

    Josh's ask — "match the actual patch, not a binary mask." A binary
    threshold erases a marker's faint anti-aliased arms (a star/plus reads as a
    blob). Weighting the colour-masked glyph by its value channel keeps that
    graded structure, so NCC can tell a plus from a filled circle at low res.
    """
    m = mask > 0
    if not m.any():
        return canonical_template(mask)
    v = hsv_up[:, :, 2].astype(np.float32) / 255.0
    return canonical_template(m.astype(np.float32) * v)


def _legend_shape_tmpl(e: LegendEntry) -> np.ndarray:
    """Intensity template for a legend marker (cached on the entry)."""
    t = getattr(e, "_shape_tmpl", _UNSET)
    if t is _UNSET:
        glyph, hue, hsv_up = _read_glyph(e.marker, sat_only=True)
        t = _intensity_template(_color_masked_glyph(glyph, hsv_up, hue), hsv_up)
        e._shape_tmpl = t
    return t


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation (mean-subtracted) of two equal-size templates."""
    av = a.ravel().astype(np.float32) - float(a.mean())
    bv = b.ravel().astype(np.float32) - float(b.mean())
    na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(av @ bv / (na * nb))


def _shape_score(dot_tmpl: np.ndarray, entry: LegendEntry) -> float:
    """Shape similarity between an aerial dot and a legend entry's marker."""
    if _SHAPE_MATCH == "cosine":
        return _template_similarity(dot_tmpl, entry.template)
    return _ncc(dot_tmpl, _legend_shape_tmpl(entry))


def _color_vec_from_glyph(
    glyph: np.ndarray, hsv_up: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Colour signature of a marker/dot from its own coloured pixels.

    Measured identically for legend markers and aerial dots so they live in the
    same space. Returns None for a grey glyph (no saturated pixels). In "lab"
    space the vector is (L, a, b) (OpenCV 8-bit); in "hsv" it is
    (S, V, cos·k, sin·k) — hue as cos/sin so red's 0/180 wrap is handled and
    the discriminative S/V axes dominate.
    """
    gm = glyph > 0
    if not gm.any():
        return None
    s = hsv_up[:, :, 1]
    v = hsv_up[:, :, 2]
    coloured = gm & (s > _SAT_MIN) & (v > _VAL_MIN)
    if int(coloured.sum()) < 3:
        return None
    if _COLOR_SPACE == "lab":
        bgr = cv2.cvtColor(hsv_up, cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return np.array([lab[:, :, c][coloured].mean() for c in range(3)],
                        dtype=np.float32)
    hue = _circular_mean_hue(hsv_up[:, :, 0][coloured])
    rad = hue * (np.pi / 90.0)
    k = 128.0
    return np.array([float(s[coloured].mean()), float(v[coloured].mean()),
                     float(np.cos(rad) * k), float(np.sin(rad) * k)],
                    dtype=np.float32)


def _color_vec_from_rgb(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Colour signature of a marker crop (used to build the legend palette)."""
    if rgb is None or rgb.size == 0:
        return None
    glyph, _hue, hsv_up = _read_glyph(rgb, sat_only=True)
    return _color_vec_from_glyph(glyph, hsv_up)


def _color_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two colour vectors (same space)."""
    return float(np.linalg.norm(a - b))


def _legend_palette(
    legend: list[LegendEntry],
) -> list[tuple[LegendEntry, Optional[np.ndarray]]]:
    """
    Each legend entry paired with its colour vector (None for grey markers).

    Cached on the entry (`_color_vec`) so repeated assign_classes calls on the
    same legend don't re-read every marker.
    """
    palette = []
    for e in legend:
        vec = getattr(e, "_color_vec", _UNSET)
        if vec is _UNSET:
            vec = _color_vec_from_rgb(e.marker)
            e._color_vec = vec
        palette.append((e, vec))
    return palette


def detect_dots(
    aerial_rgb: np.ndarray,
    exclude: Optional[tuple[int, int, int, int]] = None,
) -> list[AerialDot]:
    """
    Detect colored dots in the aerial and read each one's colour + shape.

    Merged clusters of overlapping dots are split so dense colonies are not
    undercounted. Each recovered center is read for colour and (for isolated
    dots) shape; clustered dots take the cluster's local colour.
    """
    centers, single_area = _dot_centers(aerial_rgb, exclude)
    return _dots_from_centers(aerial_rgb, centers, single_area)


def _dots_from_centers(
    aerial_rgb: np.ndarray,
    centers: list[tuple[float, float]],
    single_area: float,
) -> list[AerialDot]:
    """
    Read colour, shape and template for each dot center.

    Shared by both detectors: the colour path (`detect_dots`) finds centers by
    HSV thresholding, the subtraction path (`detect_dots_subtract`) finds them by
    image difference. Once a center is known, the per-dot feature extraction is
    identical, so `assign_classes` sees the same AerialDot shape regardless of
    how the dot was located. `single_area` sizes the crop window around each
    center — the colour path passes its median-compact estimate, the subtraction
    path its distance-transform modal marker area.
    """
    if not centers:
        return []
    half = max(4, int(round(np.sqrt(single_area) * 0.9)))
    H, W = aerial_rgb.shape[:2]

    dots: list[AerialDot] = []
    for (cx, cy) in centers:
        y0, y1 = max(0, int(cy) - half), min(H, int(cy) + half + 1)
        x0, x1 = max(0, int(cx) - half), min(W, int(cx) + half + 1)
        crop = aerial_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        glyph, hue, hsv_up = _read_glyph(crop, sat_only=True)
        color = _name_hue(hue) if hue is not None else "grey"
        # Background removal: shape/template read from the colour-masked glyph so
        # grey/off-colour background is not baked into the match signal.
        cmask = _color_masked_glyph(glyph, hsv_up, hue)
        # Shape is reliable only for isolated, marker-like glyphs.
        shape = _classify_shape(cmask) if _is_marker_like(cmask, aerial=True) else "unknown"
        template = _intensity_template(cmask, hsv_up)
        dots.append(AerialDot(
            cx=cx, cy=cy, color=color, shape=shape,
            area=int(single_area), template=template,
            quality=_dot_quality(glyph, hsv_up),
            color_vec=_color_vec_from_glyph(glyph, hsv_up),
        ))
    return dots


def detect_dots_subtract(
    screenshot_rgb: np.ndarray,
    original_rgb: np.ndarray,
    alignment: AlignResult,
    exclude: Optional[tuple[int, int, int, int]] = None,
) -> list[AerialDot]:
    """
    Detect dots by image difference (the primary path), then read each one's
    colour/shape exactly as the colour detector does.

    Subtraction against the aligned clean original locates annotation ink far
    more precisely than colour thresholds (8.40x -> 1.24x median over-detection),
    but says nothing about class. So it only supplies the centers here; the
    shared `_dots_from_centers` reads colour + template at each, and
    `assign_classes` maps them to legend classes as before. Falls back to the
    colour path when alignment was refused, which is the behaviour to ship.

    Colour is deliberately NOT used to filter the ink at this stage — that stays
    a separate, measured step, so this function's count matches standalone
    `dot_candidates` (parity) rather than quietly folding in a colour reject.
    """
    if not alignment.ok:
        return detect_dots(screenshot_rgb, exclude)

    # Lazy import: subtract imports classify (_split_cluster), so a top-level
    # import here would close the cycle.
    from src.subtract import extract_annotations, dot_candidates

    sub = extract_annotations(screenshot_rgb, original_rgb, alignment)
    cands = dot_candidates(sub)
    centers = [(cx, cy) for (cx, cy, _w, _h, _a) in cands]
    if exclude is not None:
        ex, ey, ew, eh = exclude
        centers = [(cx, cy) for (cx, cy) in centers
                   if not (ex <= cx <= ex + ew and ey <= cy <= ey + eh)]
    if not centers:
        return []
    # Single-marker crop size from the subtraction blobs themselves. dot_candidates
    # has already split merged clusters, so each candidate is ~one marker and the
    # median area is a clean per-dot estimate.
    areas = [a for (_cx, _cy, _w, _h, a) in cands if a > 0]
    single_area = max(float(np.median(areas)), 12.0) if areas else 12.0
    return _dots_from_centers(screenshot_rgb, centers, single_area)


def _dot_quality(glyph: np.ndarray, hsv_up: np.ndarray) -> float:
    """
    How 'dot-like' a detection is, for ranking before top-N selection.

    Real annotation dots are vividly saturated and bright; background texture
    (rocks, vegetation) that sneaks past the colour threshold is duller. Score
    = mean (saturation * value) over the glyph pixels, normalized to ~[0, 1].
    """
    gm = glyph > 0
    if not gm.any():
        return 0.0
    s = hsv_up[:, :, 1].astype(np.float32)[gm]
    v = hsv_up[:, :, 2].astype(np.float32)[gm]
    return float((s * v).mean() / (255.0 * 255.0))


def _lightness_offset(
    dots: list[AerialDot],
    palette: list[tuple[LegendEntry, Optional[np.ndarray]]],
) -> float:
    """
    How much darker (or lighter) this frame's aerial markers read than the legend's.

    A legend glyph is drawn crisply on a white table cell; the same marker in the
    aerial is a few pixels of thin stroke over vegetation, sand or water, at about
    a quarter of the original resolution. Anti-aliasing and downscaling mix the
    background into the marker, so its measured lightness moves toward whatever it
    sits on, while its hue survives. On `19May18Camera2-Card1-00620` the legend's
    LAGU markers read L=243 and the aerial's yellow asterisks on dark marsh land
    far below that: 152 of 183 detections were beyond `_COLOR_REJECT` with a median
    distance of 87, nearly twice the threshold, and were dropped before they could
    be classified at all.

    Measured over the nine labelled frames, the share of dots within the reject
    distance is 24/6/17/70/82/82/98/99/99% on full Lab and
    97/100/90/100/100/100/100/100/100% on chroma alone — so the drift is in
    lightness, not colour.

    The fix is a shift, not a discard. Dropping L would also erase the difference
    between a light and a dark red, which is the only thing separating two classes
    that share a hue (`scripts/diagnose_marker_colors.py`). So match on chroma,
    which is reliable, then take the MEDIAN difference in L to the matched legend
    row. The median ignores the minority of dots sitting on an unusual background,
    and a per-frame constant leaves every between-class L difference intact.

    Returns 0.0 when there is nothing to estimate from, which reproduces the old
    behaviour exactly.
    """
    vecs = [(e, v) for e, v in palette if v is not None]
    if not vecs:
        return 0.0
    P = np.array([v for _e, v in vecs], dtype=np.float32)
    diffs = []
    for d in dots:
        if d.color_vec is None:
            continue
        chroma = np.linalg.norm(P[:, 1:] - d.color_vec[1:], axis=1)
        j = int(np.argmin(chroma))
        if chroma[j] > _CHROMA_MATCH:
            continue          # colour not in this palette at all; not evidence
        diffs.append(float(d.color_vec[0] - P[j, 0]))
    if len(diffs) < _OFFSET_MIN_DOTS:
        return 0.0
    return float(np.median(diffs))


def _row_offsets(
    dots: list[AerialDot],
    palette: list[tuple[LegendEntry, Optional[np.ndarray]]],
) -> dict[int, np.ndarray]:
    """
    A per-row colour offset, estimated from the dots a first pass already assigned.

    `_lightness_offset` applies one shift to the whole frame. Measured against the
    hand labels that is the wrong shape: the drift is **per row**, and by a lot. On
    `17May10Camera2-Card1-5745` the frame-wide median is `a = -12.5`, while row 0
    needs `-29.4` and row 2 needs `-11.0`; one shared shift serves one and strands
    the other. On `19May18Camera2-Card1-00620` the same is true of lightness — row 0
    needs `L = -84.3`, row 1 needs `-16.0`.

    The mechanism is the glyph, not the class: `LAGU site` is a thin asterisk whose
    few pixels dissolve into the background, `LAGU bird` a filled circle that keeps
    its own colour. Same hue, offsets 68 apart.

    Estimating it without labels is possible because assignment already places most
    dots correctly, so the dots a row received describe that row's own drift.
    Measured against the labels, the estimate is accurate where the first pass was:
    on `5745` the median error is 3.0 across six rows, well inside the 22 candidate
    margin. Where the first pass was wrong it is wrong too — `00620` row 1 received
    30 dots of which only 20 belong, and its estimate lands 76.7 away, describing row
    0's drift rather than its own.

    That failure is why the caller only ever **adds** candidates with these offsets
    and never removes one (see `_color_candidates`). A contaminated offset can then
    do no more than offer an extra row for the template score to reject; it cannot
    take away a row the frame-wide shift already found.

    Discarding the contaminated estimates was tried and is **worse**. When a row
    absorbs another's dots the two estimates converge — on `00620` rows 0 and 1 land
    within 3 of each other while their true offsets are 68 apart — so dropping any
    row whose estimate sits on a larger row's looked principled. Measured, it saved
    one dot on `00620` and cost six on `5745`: pooled 0.789 → 0.779. A contaminated
    offset is only ever an extra candidate, and the template score handles it; the
    guard instead removed offsets that were doing real work.

    Returns `{row: [dL, da, db]}`, skipping rows with too few dots to take a median.
    """
    if not _ROW_OFFSET:
        return {}
    vec = {e.row: v for e, v in palette if v is not None}
    got: dict[int, list[np.ndarray]] = {}
    for d in dots:
        if d.legend_row is None or d.color_vec is None:
            continue
        v = vec.get(d.legend_row)
        if v is not None:
            got.setdefault(d.legend_row, []).append(d.color_vec - v)
    return {r: np.median(np.array(g), axis=0).astype(np.float32)
            for r, g in got.items() if len(g) >= _OFFSET_MIN_DOTS}


def _color_candidates(
    d: AerialDot, palette: list[tuple[LegendEntry, Optional[np.ndarray]]],
    l_offset: float = 0.0,
    row_offsets: Optional[dict[int, np.ndarray]] = None,
) -> list[LegendEntry]:
    """
    Legend entries whose colour matches the dot's, via the dialog's own palette.

    A dot with no colour (grey) matches grey legend entries. Otherwise we take
    every entry within _COLOR_MARGIN of the closest one — a self-adapting
    colour cluster: same-colour classes stay together, a far light/dark variant
    drops out. Returns [] (dot left unassigned) when the closest legend colour
    is beyond _COLOR_REJECT (off-palette noise).

    `l_offset` shifts the legend's lightness onto this frame's aerial scale; see
    `_lightness_offset`. It is a constant per frame, so the reject still rejects —
    a dot of a colour the legend does not contain stays far away on the chroma
    axes, which the shift does not touch.

    `row_offsets` is the second pass (see `_row_offsets`). Each row also gets a
    chance to claim the dot under its **own** measured drift, and any row that does
    is **added** to the candidate list. Nothing is ever removed by this: a row the
    frame-wide shift already found stays a candidate whatever its own offset says.
    That containment is deliberate — the per-row estimate is accurate where the
    first pass was right (median error 3.0 on `5745`) and badly wrong where it was
    not (76.7 on one row of `00620`), so a bad estimate must be unable to do more
    than offer an extra row for the template score to reject.
    """
    if not _COLOR_ANCHOR:
        # Ablation baseline: old global colour-NAME grouping (pre-rework).
        return [e for e, _v in palette if e.color == d.color]
    if d.color_vec is None:
        return [e for e, v in palette if v is None]
    shift = np.array([l_offset, 0.0, 0.0], dtype=np.float32)
    dists = [(e, _color_dist(d.color_vec, v + shift))
             for e, v in palette if v is not None]
    if not dists:
        return []
    dmin = min(dd for _, dd in dists)
    out = [e for e, dd in dists if dd <= dmin + _COLOR_MARGIN] if dmin <= _COLOR_REJECT \
        else []
    if not row_offsets:
        return out
    have = {id(e) for e in out}
    for e, v in palette:
        if v is None or id(e) in have:
            continue
        off = row_offsets.get(e.row)
        if off is not None and _color_dist(d.color_vec, v + off) <= _COLOR_MARGIN:
            out.append(e)
    return out


def _pair_score(d: AerialDot, e: LegendEntry, sole: bool,
                color_dist: Optional[float] = None) -> float:
    """
    How well a dot matches one legend row.

    Always the intensity-template NCC, optionally boosted when the dot's own
    classified shape name agrees with the row's (off by default; the coarse shape
    label reinforced wrong classes).

    ## Why colour is in the score at all

    Colour used to only *gate* candidacy: a dot either reached a row's margin or it
    did not, and after that the template NCC ranked alone. So a dot sitting exactly
    on a row's colour and one that scraped in at the edge of the margin competed as
    equals for the same slot. With `_row_offsets` giving a per-row colour distance
    worth trusting, that distance can rank as well as gate. `color_dist` is folded
    in as `1 - dist/_COLOR_REJECT`, weighted by `_SCORE_COLOR`; passing None leaves
    the score exactly as it was.

    A dot whose colour picked out a single row used to score a flat 1.0, which put
    it ahead of every contested pair in the capacity queue. That conflated two
    different things: having one candidate says the legend holds only one class of
    that colour, not that this dot is a good marker. Measured over the four
    labelled frames, those dots are **less** likely to be real than contested ones —
    69% against 80% — yet they were taking the slots first, and 43 genuine markers
    arrived at a row that was already full. Scoring every pair the same way puts
    them on one scale, so a slot goes to the dot that actually matches best.
    `SOLE_CANDIDATE_WINS=1` restores the old behaviour for A/B.
    """
    if sole and _SOLE_WINS:
        return 1.0
    s = _shape_score(d.template, e) + (_SHAPE_BOOST if d.shape == e.shape else 0.0)
    if _SCORE_COLOR and color_dist is not None:
        s += _SCORE_COLOR * max(0.0, 1.0 - color_dist / _COLOR_REJECT)
    return s


def _uncounted_capacity(legend: list[LegendEntry]) -> dict[int, int]:
    """
    Capacity for rows whose Count column never read — `{id(entry): capacity}`.

    Rows missing from the result stay unlimited, which is what the code did
    originally, and it is the wrong default for the one case that matters. A row
    parsed off the bottom of the table — the horizontal scrollbar, or the photo
    below the dialog — has no marker and no count, so it looks exactly like a row
    whose Count merely failed to read. Being unlimited, it then absorbs dots
    belonging to real rows: on `5745` two such rows take 49 dots, 31 of them real
    `LAGU sit`, while row 0 fills to 81 of its stated 150.

    The opposite risk is equally real. `_ocr_count` returns `None` whenever the cell
    held dark pixels it could not resolve into digits, which happens on genuine
    rows, and zeroing those destroys a frame: on `238` (a frame selection rejects,
    where only 1 row in 7 reads a count) per-dot accuracy fell 0.667 → 0.018.

    `"tail"` is the narrow rule that separates the two: only rows *below the last
    row that did read a count* are treated as outside the table. A real row whose
    OCR failed sits among counted rows and is left alone.

    Its limit is worth stating: `tail` is only as good as the counted rows' reach.
    On `238` just 3 of 13 rows read a count and the last of them is row 7, so five
    real rows below it are zeroed. No selected frame is anywhere near that — the
    weakest reads 3 of 4 — but a frame whose Count column barely reads will be
    damaged rather than helped.
    """
    if _UNCOUNTED not in ("zero", "tail", "share"):
        return {}
    unread = [e for e in legend if e.count is None]
    if not unread:
        return {}
    if _UNCOUNTED == "zero":
        return {id(e): 0 for e in unread}
    if _UNCOUNTED == "tail":
        last = max((e.row for e in legend if e.count is not None), default=None)
        if last is None:
            return {}
        return {id(e): 0 for e in unread if e.row > last}
    # "share": hand the unread rows only what the dialog's own total leaves over.
    # That total reads on 11 of 25 frames, so this stays off unless it is present.
    total = getattr(legend[0], "dialog_total", None)
    if total is None:
        return {}
    counted = sum(int(e.count) for e in legend if e.count is not None)
    spare = max(0, int(total) - counted) // len(unread)
    return {id(e): spare for e in unread}


def assign_classes(
    dots: list[AerialDot], legend: list[LegendEntry],
    respect_counts: bool = _RESPECT_COUNTS,
) -> list[AerialDot]:
    """
    Assign each dot to a legend row: colour first (anchored to the dialog's own
    palette), then shape (template) among the same-colour candidates, and finally
    under each row's stated Count as a **capacity**. Mutates and returns `dots`.

    ## Why capacity

    Assigning each dot independently to its own best row ignores what the dialog
    says about how many dots each row can own, and the result contradicts the
    legend wholesale. On `17May10Camera2-Card1-5745`, measured against 372 hand
    labels:

        row          Count   assigned   labelled
        LAGU sit       150         13        139
        LAGU stand       0         59          0
        ROSP site       62         56         60
        ROSP bird        0         76          0
        WHIB site      106         20        103
        WHIB bird        2         37          2

    Four rows state a count of zero and were given 156 dots; 231 of 384 assignments
    sat beyond their row's stated count. The failure is systematic rather than
    noisy — populous `site` rows are left nearly empty while the empty `bird` and
    `stand` rows fill up, which is the label swap `learnings.md` #30 describes.

    So instead of per-dot argmax we score every (dot, candidate row) pair, sort by
    score, and assign best-first, skipping rows that are full. A displaced dot
    falls through to its next-best row rather than being dropped, because that
    pair is still in the queue. A dot whose every candidate row is full is left
    unassigned, which is the right reading: the legend says those rows are
    accounted for, so the surplus detection is most likely background.

    Note the difference from `select_by_count`, which caps a group at its top N
    **without reassigning** — that removes the surplus but never moves a dot to
    where it belongs, so it cannot undo a swap.

    ## Why this is not the retracted count prior

    That experiment fed **survey CSV** counts into the pipeline and then scored
    against those same counts, which is circular, and it broke the rule that the
    pipeline reads only the image. Here the counts come from the **dialog's own
    Count column**, which `legend.attach_class_names` already parses from the
    screenshot, and the result is scored against **hand-labelled dots**. Different
    input, independent scorer.

    ## Fallbacks

    Count OCR reads roughly 60-65% of cells, so a missing count must never be read
    as zero: a row whose `count` is None is uncapped and behaves exactly as before.
    A legend with no readable counts at all therefore reproduces the old behaviour
    dot for dot. Set `respect_counts=False` (or `RESPECT_COUNTS=0`) to force that.
    """
    palette = _legend_palette(legend)
    # One shift for the whole frame, measured from the frame's own dots.
    l_offset = _lightness_offset(dots, palette) if _L_OFFSET else 0.0

    # Capacity where the dialog gave a number, plus whatever `_UNCOUNTED` says a
    # row with no readable count may hold.
    capacity = {}
    if respect_counts:
        # A row cannot hold more dots than the detector found on the whole frame, so
        # a count above that is a misread and worse than no count at all: capacity
        # caps at it, the cap never binds, and the row absorbs whatever it likes.
        # Measured over the 25 selected frames, three read one such row each —
        # `0537` row 0 at 1237 against 363 detections, `0216` row 1 at 2301 against
        # 1174, `06389` row 0 at 861 against 74 — and in every case the inflation is
        # in a single row, 74-99% of the frame's whole count sum. No frame that reads
        # sanely has any row over the bound, so this touches only the broken ones.
        # Treating it as unread hands the row back to `_uncounted_capacity`.
        limit = len(dots) if _COUNT_SANITY else float("inf")
        for e in legend:
            if e.count is not None and 0 <= e.count <= limit:
                capacity[id(e)] = int(e.count)
        capacity.update(_uncounted_capacity(legend))

    def _assign(row_offsets, retry_blocked=True):
        """
        One full pass: score every candidate pair, then fill best-first.

        `retry_blocked` is off for the first pass. That pass exists only to estimate
        each row's colour drift (`_row_offsets`), so it has to stay conservative —
        letting it place blocked dots too changed which dots each row held, moved the
        offsets, and cost `00620` five dots in the pass that actually counts.
        """
        # `quality` breaks ties so a contested row goes to the more marker-like
        # dot, and keeps the order deterministic.
        vec = {e.row: v for e, v in palette if v is not None}
        frame_shift = np.array([l_offset, 0.0, 0.0], dtype=np.float32)

        def dist(d, e):
            v = vec.get(e.row)
            if v is None or d.color_vec is None:
                return None
            off = (row_offsets or {}).get(e.row)   # the row's own drift, else the frame's
            return _color_dist(d.color_vec, v + (frame_shift if off is None else off))

        def scored(d, i, cands):
            sole = len(cands) == 1
            return [(_pair_score(d, e, sole, dist(d, e)), d.quality, i, e)
                    for e in cands]

        cands = [_color_candidates(d, palette, l_offset, row_offsets) for d in dots]
        pairs: list[tuple[float, float, int, LegendEntry]] = []
        for i, d in enumerate(dots):
            pairs.extend(scored(d, i, cands[i]))
        pairs.sort(key=lambda t: (-t[0], -t[1]))
        used: dict[int, int] = {}
        taken: dict[int, tuple[LegendEntry, float]] = {}
        for score, _q, i, e in pairs:
            if i in taken:
                continue
            cap = capacity.get(id(e))
            if cap is not None and used.get(id(e), 0) >= cap:
                continue
            taken[i] = (e, score)
            used[id(e)] = used.get(id(e), 0) + 1

        # A dot whose every candidate row filled up is currently dropped. It already
        # falls through to its next-best CANDIDATE, so this only catches the dot with
        # nowhere left to fall. On `5745` that is 25 detections whose one candidate is
        # `ROSP bird`, a row the dialog genuinely counts as 0 — the count is right, the
        # candidate set was too narrow to also offer `ROSP site` next door.
        #
        # The retry widens to every row still inside `_COLOR_REJECT`, which is the
        # palette's existing "not this colour at all" boundary, not a new threshold.
        # Only blocked dots are reconsidered, so no dot that already has a row is put
        # at risk.
        if _BLOCKED_RETRY and retry_blocked:
            spare = [e for e, v in palette
                     if v is not None and
                     (capacity.get(id(e)) is None or
                      used.get(id(e), 0) < capacity[id(e)])]
            if spare:
                retry: list[tuple[float, float, int, LegendEntry]] = []
                for i, d in enumerate(dots):
                    # `not cands[i]` means colour rejected the dot outright — it is
                    # off-palette, which is the valid/invalid decision doing its job,
                    # and forcing it onto a row costs more than it recovers (measured
                    # on `00620`: -5 dots when those were included, +0 when not).
                    if i in taken or d.color_vec is None or not cands[i]:
                        continue
                    near = [e for e in spare
                            if (dd := dist(d, e)) is not None and dd <= _COLOR_REJECT]
                    retry.extend(scored(d, i, near))
                retry.sort(key=lambda t: (-t[0], -t[1]))
                for score, _q, i, e in retry:
                    if i in taken:
                        continue
                    cap = capacity.get(id(e))
                    if cap is not None and used.get(id(e), 0) >= cap:
                        continue
                    taken[i] = (e, score)
                    used[id(e)] = used.get(id(e), 0) + 1
        return pairs, taken

    def _write(taken):
        for i, d in enumerate(dots):
            best = taken.get(i)
            if best is None:
                d.legend_row = d.species = d.category = d.class_name = None
                d.match_score = None
                continue
            e, score = best
            d.legend_row = e.row
            d.species, d.category = e.species, e.category
            d.class_name = e.class_name
            d.match_score = round(score, 3)

    # First pass exists to measure, not to decide: it places dots so each row's own
    # colour drift can be read off them, and it stays conservative so that reading
    # is not skewed by dots the retry would have forced somewhere.
    pairs, taken = _assign(None, retry_blocked=False)
    if not pairs:
        return dots
    _write(taken)

    # Final pass. The drift from legend glyph to aerial marker is per row, not per
    # frame (`_row_offsets`), and those offsets only ADD candidates, so no dot the
    # first pass placed is put at risk. This pass always runs — it also carries the
    # blocked-dot retry, which must not depend on whether any row had enough dots
    # to yield an offset.
    offsets = _row_offsets(dots, palette) if _ROW_OFFSET else {}
    _, final = _assign(offsets)
    _write(final)
    return dots


def legend_key(e: LegendEntry) -> int:
    """The grouping key for a legend row. Use this to build `expected` counts."""
    return e.row


def _dot_key(d: AerialDot):
    """
    Grouping key for a dot's class.

    The legend ROW is the identity, not the name. Class names come from OCR and
    are neither unique nor always present: one screenshot parses two rows both
    named 'ad', another two both named 'BRPE wbn', and rows whose name never read
    carry None. Keying by name merged those into one group, so `select_by_count`
    ran top-N over two classes at once and capped the wrong dots. Falls back to
    the name, then colour/shape, for dots that were never assigned a row.
    """
    if d.legend_row is not None:
        return d.legend_row
    return d.class_name or f"{d.color}/{d.shape}"


def class_counts(dots: list[AerialDot]) -> dict:
    """
    Per-class dot counts, keyed by legend row index for assigned dots.

    Unassigned dots fall back to their class name, then to 'colour/shape', so the
    keys are mixed int/str by design — see `_dot_key`.
    """
    counts: dict = {}
    for d in dots:
        counts[_dot_key(d)] = counts.get(_dot_key(d), 0) + 1
    return counts


def select_by_count(
    dots: list[AerialDot],
    expected: dict,
    keep_unknown: bool = True,
    quality_min: float = 0.0,
) -> list[AerialDot]:
    """
    Count-guided precision filter: keep the top-N highest-quality dots per
    class, where N is the expected count for that class.

    This removes background false positives and over-split duplicates on noisy
    images — the genuine dots score highest on `quality`, so the top-N survive.
    `expected` maps class key -> count (from the legend's own Count column, so
    the detector stays self-contained). Build it with `legend_key`:

        expected = {legend_key(e): e.count for e in entries if e.count is not None}

    Keying by row rather than by name matters here: two rows sharing an OCR'd
    name used to land in one group, so top-N was applied to the pair at once and
    capped the wrong dots. Classes absent from `expected` are kept in full when
    keep_unknown is True, dropped otherwise. `quality_min` drops low-quality dots
    up front (count-free noise floor for classes without a known count).
    """
    groups: dict = {}
    for d in dots:
        groups.setdefault(_dot_key(d), []).append(d)

    kept: list[AerialDot] = []
    for key, group in groups.items():
        if key in expected:
            n = max(0, int(expected[key]))
            group.sort(key=lambda d: d.quality, reverse=True)
            kept.extend(group[:n])
        elif keep_unknown:
            kept.extend(d for d in group if d.quality >= quality_min)
    return kept
