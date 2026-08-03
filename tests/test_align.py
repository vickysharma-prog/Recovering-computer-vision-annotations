from __future__ import annotations

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.align import align, warp_to_screenshot


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _textured_original(w: int = 1200, h: int = 800, seed: int = 0) -> np.ndarray:
    """A synthetic aerial-like scene with enough structure for SIFT.

    Blobs of varying size and colour on a noisy ground, which is what the real
    imagery looks like to a feature detector: many small distinctive corners.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 150, (h, w, 3), dtype=np.uint8)
    for _ in range(260):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        colour = tuple(int(v) for v in rng.integers(30, 230, 3))
        cv2.circle(img, c, int(rng.integers(4, 16)), colour, -1)
    for _ in range(40):
        x, y = int(rng.integers(0, w - 90)), int(rng.integers(0, h - 90))
        colour = tuple(int(v) for v in rng.integers(30, 230, 3))
        cv2.rectangle(img, (x, y), (x + 70, y + 55), colour, -1)
    return img


def _screenshot_from(original: np.ndarray, scale: float = 0.5,
                     dx: int = 12, dy: int = 20) -> np.ndarray:
    """Emulate how a screenshot relates to its original: downscaled + offset."""
    small = cv2.resize(original, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    canvas = np.zeros((h + dy, w + dx, 3), np.uint8)
    canvas[dy:, dx:] = small
    return canvas


# ─────────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────────

class TestAlign:
    def test_recovers_scale_and_offset(self):
        original = _textured_original()
        shot = _screenshot_from(original, scale=0.5, dx=12, dy=20)

        res = align(shot, original)

        assert res.ok, res.reason
        assert res.reproj_err < 3.0
        assert res.inliers >= 10

    def test_prefers_similarity_when_there_is_no_perspective(self):
        """Real pairs are scale+translation, so the simpler model should win."""
        original = _textured_original(seed=3)
        shot = _screenshot_from(original, scale=0.6, dx=8, dy=8)

        assert align(shot, original).model == "similarity"

    def test_transform_maps_a_known_point(self):
        original = _textured_original(seed=5)
        scale, dx, dy = 0.5, 12, 20
        shot = _screenshot_from(original, scale=scale, dx=dx, dy=dy)
        res = align(shot, original)
        assert res.ok

        # A point in the screenshot maps back to work-scale original coords.
        pt = np.float32([[[400.0, 300.0]]])
        got = cv2.perspectiveTransform(pt, res.H)[0][0]
        expected_full = ((400.0 - dx) / scale, (300.0 - dy) / scale)
        expected = (expected_full[0] * res.scale, expected_full[1] * res.scale)
        assert got[0] == pytest.approx(expected[0], abs=6)
        assert got[1] == pytest.approx(expected[1], abs=6)

    def test_rejects_unrelated_images(self):
        """A silent bad warp is worse than a refusal at corpus scale."""
        rng = np.random.default_rng(1)
        res = align(_textured_original(seed=7),
                    rng.integers(0, 255, (700, 900, 3), dtype=np.uint8))

        assert not res.ok
        assert res.H is None
        assert res.reason

    def test_rejects_empty_input(self):
        assert not align(np.zeros((0, 0, 3), np.uint8),
                         _textured_original()).ok

    def test_flat_image_has_no_features(self):
        """Open water is textureless; it must fail rather than fit noise."""
        flat = np.full((600, 800, 3), 120, np.uint8)
        assert not align(flat, flat).ok


# ─────────────────────────────────────────────────
# WARPING
# ─────────────────────────────────────────────────

class TestWarp:
    def test_warp_matches_the_screenshot(self):
        original = _textured_original(seed=11)
        shot = _screenshot_from(original)
        res = align(shot, original)
        assert res.ok

        warped, covered = warp_to_screenshot(original, res, shot.shape)

        assert warped.shape == shot.shape
        assert covered.shape == shot.shape[:2]
        # Where the original reaches, the two should agree closely — that is the
        # whole premise of detecting annotations by difference.
        inside = covered > 0
        assert inside.mean() > 0.5
        delta = np.abs(warped[inside].astype(float) - shot[inside].astype(float))
        assert delta.mean() < 12.0

    def test_uncovered_region_is_marked(self):
        """The offset strip has no counterpart and must be excluded."""
        original = _textured_original(seed=13)
        shot = _screenshot_from(original, dx=40, dy=40)
        res = align(shot, original)
        assert res.ok

        _warped, covered = warp_to_screenshot(original, res, shot.shape)
        assert covered[:5, :].mean() < 128

    def test_refuses_to_warp_a_rejected_alignment(self):
        rng = np.random.default_rng(2)
        bad = align(_textured_original(seed=17),
                    rng.integers(0, 255, (600, 800, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            warp_to_screenshot(_textured_original(), bad, (600, 800, 3))
