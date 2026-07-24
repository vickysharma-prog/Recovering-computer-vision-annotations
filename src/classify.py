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
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter

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
    species: Optional[str] = None
    category: Optional[str] = None
    class_name: Optional[str] = None
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


def _color_candidates(
    d: AerialDot, palette: list[tuple[LegendEntry, Optional[np.ndarray]]],
) -> list[LegendEntry]:
    """
    Legend entries whose colour matches the dot's, via the dialog's own palette.

    A dot with no colour (grey) matches grey legend entries. Otherwise we take
    every entry within _COLOR_MARGIN of the closest one — a self-adapting
    colour cluster: same-colour classes stay together, a far light/dark variant
    drops out. Returns [] (dot left unassigned) when the closest legend colour
    is beyond _COLOR_REJECT (off-palette noise).
    """
    if not _COLOR_ANCHOR:
        # Ablation baseline: old global colour-NAME grouping (pre-rework).
        return [e for e, _v in palette if e.color == d.color]
    if d.color_vec is None:
        return [e for e, v in palette if v is None]
    dists = [(e, _color_dist(d.color_vec, v)) for e, v in palette if v is not None]
    if not dists:
        return []
    dmin = min(dd for _, dd in dists)
    if dmin > _COLOR_REJECT:
        return []
    return [e for e, dd in dists if dd <= dmin + _COLOR_MARGIN]


def assign_classes(
    dots: list[AerialDot], legend: list[LegendEntry],
) -> list[AerialDot]:
    """
    Assign each dot to a legend class: colour first (anchored to the dialog's
    own palette), then shape (template) to break ties among same-colour classes.
    Mutates and returns `dots`.
    """
    palette = _legend_palette(legend)

    for d in dots:
        cands = _color_candidates(d, palette)
        if not cands:
            continue  # off-palette / colour absent -> leave unassigned
        if len(cands) == 1:
            best, score = cands[0], 1.0
        else:
            # Intensity-template NCC, boosted when the dot's own classified
            # shape name agrees with the candidate's (a strong independent
            # signal). Both are tunable via env for ablation.
            scored = [
                (e, _shape_score(d.template, e)
                 + (_SHAPE_BOOST if d.shape == e.shape else 0.0))
                for e in cands
            ]
            best, score = max(scored, key=lambda t: t[1])
        d.species, d.category = best.species, best.category
        d.class_name = best.class_name
        d.match_score = round(score, 3)
    return dots


def _dot_key(d: AerialDot) -> str:
    return d.class_name or f"{d.color}/{d.shape}"


def class_counts(dots: list[AerialDot]) -> dict[str, int]:
    """Per-class dot counts (keyed by class_name, falling back to colour/shape)."""
    counts: dict[str, int] = {}
    for d in dots:
        counts[_dot_key(d)] = counts.get(_dot_key(d), 0) + 1
    return counts


def select_by_count(
    dots: list[AerialDot],
    expected: dict[str, int],
    keep_unknown: bool = True,
    quality_min: float = 0.0,
) -> list[AerialDot]:
    """
    Count-guided precision filter: keep the top-N highest-quality dots per
    class, where N is the expected count for that class.

    This removes background false positives and over-split duplicates on noisy
    images — the genuine dots score highest on `quality`, so the top-N survive.
    `expected` maps class key -> count (from the legend's own Count column, so
    the detector stays self-contained). Classes absent from `expected` are kept
    in full when keep_unknown is True, dropped otherwise. `quality_min` drops
    low-quality dots up front (count-free noise floor for classes without a
    known count).
    """
    groups: dict[str, list[AerialDot]] = {}
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
