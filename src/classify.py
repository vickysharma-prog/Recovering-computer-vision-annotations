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
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter

from src.legend import (
    LegendEntry,
    _read_glyph,
    _classify_shape,
    _is_marker_like,
    _name_hue,
    canonical_template,
)

logger = logging.getLogger(__name__)


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
    return centers, single_area


def _template_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two canonical glyph templates (shape match)."""
    av, bv = a.ravel(), b.ravel()
    na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(av @ bv / (na * nb))


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
        # Shape is reliable only for isolated, marker-like glyphs.
        shape = _classify_shape(glyph) if _is_marker_like(glyph) else "unknown"
        template = canonical_template(glyph)
        dots.append(AerialDot(
            cx=cx, cy=cy, color=color, shape=shape,
            area=int(single_area), template=template,
            quality=_dot_quality(glyph, hsv_up),
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


def assign_classes(
    dots: list[AerialDot], legend: list[LegendEntry],
) -> list[AerialDot]:
    """
    Assign each dot to a legend class: colour first, shape (template) to break
    ties among same-colour classes. Mutates and returns `dots`.
    """
    by_color: dict[str, list[LegendEntry]] = {}
    for e in legend:
        by_color.setdefault(e.color, []).append(e)

    for d in dots:
        cands = by_color.get(d.color)
        if not cands:
            continue  # colour not in legend -> leave unassigned
        if len(cands) == 1:
            best, score = cands[0], 1.0
        else:
            # Template correlation, boosted when the dot's own classified shape
            # name agrees with the candidate's (a strong independent signal).
            scored = [
                (e, _template_similarity(d.template, e.template)
                 + (0.35 if d.shape == e.shape else 0.0))
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
