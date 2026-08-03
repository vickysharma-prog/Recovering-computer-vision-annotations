"""
Align an annotated screenshot to its clean high-resolution original.

Every screenshot in the survey has a corresponding original photograph without
the counting tool's dots baked in. Registering the two lets a later stage isolate
the annotations by DIFFERENCE rather than by colour thresholds — colour bins
overfit one survey year and flood another, because the symbology changes.

Measured behaviour on real pairs: the screenshot's aerial region is the whole
original downscaled ~4x, with a small offset for window chrome. The recovered
transform is essentially pure scale + translation (perspective terms ~0), so a
similarity model is preferred when it fits as well as a homography — fewer
degrees of freedom is more stable on low-feature scenes such as open water or
bare sand.

The quality gate is the important part. At 18k images a silently bad warp is far
worse than a refusal: it would light up the entire frame as "annotation". So
`align()` returns `ok=False` rather than a transform it cannot vouch for, and
callers are expected to fall back to colour-based detection in that case.

Usage:
    res = align(screenshot_rgb, original_rgb)
    if res.ok:
        warped, covered = warp_to_screenshot(original_rgb, res, screenshot_rgb.shape)
"""

from __future__ import annotations

import logging
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
    return full.get("align", {})


_CONFIG = _load_config()

# Defaults keep the module usable even if config.yaml has no `align:` block.
_WORK_WIDTH = _CONFIG.get("work_width", 1600)      # originals are ~4752px wide
_MAX_FEATURES = _CONFIG.get("max_features", 8000)
_RATIO = _CONFIG.get("ratio_test", 0.75)           # Lowe's ratio
_RANSAC_THRESH = _CONFIG.get("ransac_thresh", 5.0)
_MIN_MATCHES = _CONFIG.get("min_matches", 12)
_MIN_INLIERS = _CONFIG.get("min_inliers", 10)
_MIN_INLIER_FRAC = _CONFIG.get("min_inlier_frac", 0.30)
_MAX_REPROJ = _CONFIG.get("max_reproj_err", 3.0)   # px, in work-scale units
# Below this, the homography's perspective row is indistinguishable from zero and
# the extra freedom only adds variance, so the similarity fit is used instead.
_PERSPECTIVE_EPS = _CONFIG.get("perspective_eps", 1e-5)


# ─────────────────────────────────────────────────────
# DATACLASS
# ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AlignResult:
    """Transform mapping screenshot pixels -> work-scale original pixels.

    `ok` is the only thing callers should branch on. When it is False the other
    fields are still populated for diagnostics, but `H` must not be used.
    """

    H: Optional[np.ndarray]
    scale: float          # work-scale original = original * scale
    matches: int
    inliers: int
    reproj_err: float
    model: str            # "similarity" | "homography" | "none"
    ok: bool
    reason: str = ""

    @property
    def inlier_frac(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


def _fail(reason: str, matches: int = 0, inliers: int = 0) -> AlignResult:
    logger.debug("alignment rejected: %s", reason)
    return AlignResult(None, 1.0, matches, inliers, float("inf"), "none", False, reason)


# ─────────────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────────────

def _reproj_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    projected = cv2.perspectiveTransform(src, H)
    return float(np.linalg.norm(projected - dst, axis=-1).mean())


def align(screenshot_rgb: np.ndarray, original_rgb: np.ndarray) -> AlignResult:
    """Register a screenshot against its clean original.

    The original is downscaled to `_WORK_WIDTH` first: SIFT on a 3168x4752 image
    is slow and buys nothing, since the screenshot is far lower resolution anyway.

    No attempt is made to mask the dialog beforehand. Its features have no
    counterpart in the original, so they cannot fit a consistent transform and
    RANSAC discards them as outliers — which is more robust than depending on the
    dialog locator, whose own reliability is limited.
    """
    if screenshot_rgb.size == 0 or original_rgb.size == 0:
        return _fail("empty image")

    scale = _WORK_WIDTH / original_rgb.shape[1]
    work = cv2.resize(original_rgb, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)

    sift = cv2.SIFT_create(nfeatures=_MAX_FEATURES)
    kp_shot, des_shot = sift.detectAndCompute(
        cv2.cvtColor(screenshot_rgb, cv2.COLOR_RGB2GRAY), None)
    kp_work, des_work = sift.detectAndCompute(
        cv2.cvtColor(work, cv2.COLOR_RGB2GRAY), None)
    if des_shot is None or des_work is None or len(kp_shot) < _MIN_MATCHES \
            or len(kp_work) < _MIN_MATCHES:
        return _fail("too few keypoints")

    pairs = cv2.BFMatcher().knnMatch(des_shot, des_work, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < _RATIO * n.distance]
    if len(good) < _MIN_MATCHES:
        return _fail(f"only {len(good)} ratio-test matches", len(good))

    src = np.float32([kp_shot[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_work[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_THRESH)
    if H is None or mask is None:
        return _fail("homography did not converge", len(good))

    keep = mask.ravel().astype(bool)
    inliers = int(keep.sum())
    if inliers < _MIN_INLIERS or inliers / len(good) < _MIN_INLIER_FRAC:
        return _fail(f"weak consensus ({inliers}/{len(good)} inliers)",
                     len(good), inliers)

    src_in, dst_in = src[keep], dst[keep]
    err = _reproj_error(H, src_in, dst_in)
    model = "homography"

    # The measured transform on real pairs is scale+translation. If the
    # perspective row is negligible, refit as a similarity on the inliers: it
    # cannot warp pathologically on scenes with little texture.
    if max(abs(H[2, 0]), abs(H[2, 1])) < _PERSPECTIVE_EPS:
        A, _ = cv2.estimateAffinePartial2D(src_in, dst_in, method=cv2.RANSAC,
                                           ransacReprojThreshold=_RANSAC_THRESH)
        if A is not None:
            S = np.vstack([A, [0.0, 0.0, 1.0]])
            err_sim = _reproj_error(S, src_in, dst_in)
            # Accept the simpler model unless it is clearly worse.
            if err_sim <= err * 1.25:
                H, err, model = S, err_sim, "similarity"

    if err > _MAX_REPROJ:
        return _fail(f"reprojection error {err:.2f}px", len(good), inliers)

    return AlignResult(H, scale, len(good), inliers, err, model, True)


def warp_to_screenshot(original_rgb: np.ndarray, res: AlignResult,
                       shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Warp the clean original into the screenshot's frame.

    Returns `(warped_rgb, covered)` where `covered` is a uint8 mask marking which
    screenshot pixels the original actually reaches. Anything outside it (window
    chrome, desktop, letterboxing) has no counterpart and must be excluded from
    any difference, or it registers as one enormous annotation.
    """
    if not res.ok or res.H is None:
        raise ValueError("cannot warp with a rejected alignment")

    work = cv2.resize(original_rgb, None, fx=res.scale, fy=res.scale,
                      interpolation=cv2.INTER_AREA)
    inv = np.linalg.inv(res.H)
    h, w = shape[:2]
    warped = cv2.warpPerspective(work, inv, (w, h))
    covered = cv2.warpPerspective(np.full(work.shape[:2], 255, np.uint8), inv, (w, h))
    return warped, covered
