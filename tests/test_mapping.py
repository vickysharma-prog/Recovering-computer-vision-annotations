from __future__ import annotations

from types import SimpleNamespace

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.align import AlignResult, align
from src.mapping import map_dots, to_original

from tests.test_align import _screenshot_from, _textured_original


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _result(scale: float = 1.0, sx: float = 1.0, tx: float = 0.0,
            ty: float = 0.0) -> AlignResult:
    """An accepted alignment with a hand-built scale+translation transform.

    `sx` is the screenshot -> work-scale factor and `scale` the original -> work-scale
    factor, which are two different things. Keeping them separate in the fixture is
    the point: a test that sets both to 1 cannot see the bug this module exists for.
    """
    H = np.array([[sx, 0.0, tx],
                  [0.0, sx, ty],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return AlignResult(H=H, scale=scale, matches=100, inliers=95,
                       reproj_err=0.2, model="similarity", ok=True)


def _dot(cx: float, cy: float) -> SimpleNamespace:
    """Anything carrying cx/cy maps; `classify.AerialDot` is the live case."""
    return SimpleNamespace(cx=cx, cy=cy)


# ─────────────────────────────────────────────────
# TO_ORIGINAL
# ─────────────────────────────────────────────────

class TestToOriginal:
    def test_identity_returns_the_input(self):
        pts = [(10.0, 20.0), (300.5, 400.25)]

        got = to_original(pts, _result())

        assert got == pytest.approx(np.array(pts))

    def test_applies_scale_and_translation(self):
        got = to_original([(100.0, 50.0)], _result(sx=2.0, tx=-7.0, ty=3.0))

        assert got[0] == pytest.approx((193.0, 103.0))

    def test_divides_by_the_alignment_scale(self):
        """The transform lands on the WORK-SCALE original, not the original.

        `H` alone would give (200, 100) here. The original was downscaled by 0.5
        before SIFT ran, so the full-resolution answer is twice that. Omitting the
        divide returns coordinates that are internally consistent and wrong by a
        factor of `scale`, which nothing downstream would raise on.
        """
        res = _result(scale=0.5, sx=2.0)

        got = to_original([(100.0, 50.0)], res)

        assert got[0] == pytest.approx((400.0, 200.0))

    def test_empty_input_returns_an_empty_array(self):
        got = to_original([], _result())

        assert got.shape == (0, 2)

    def test_rejects_a_failed_alignment(self):
        bad = AlignResult(H=None, scale=1.0, matches=0, inliers=0,
                          reproj_err=float("inf"), model="none", ok=False,
                          reason="too few keypoints")

        with pytest.raises(ValueError):
            to_original([(1.0, 1.0)], bad)

    def test_rejects_a_nonpositive_scale(self):
        with pytest.raises(ValueError):
            to_original([(1.0, 1.0)], _result(scale=0.0))


# ─────────────────────────────────────────────────
# MAP_DOTS
# ─────────────────────────────────────────────────

class TestMapDots:
    def test_keeps_both_coordinate_pairs_and_the_order(self):
        dots = [_dot(10.0, 20.0), _dot(30.0, 40.0)]

        got = map_dots(dots, _result(scale=0.5), (2000, 3000, 3))

        assert [m.index for m in got] == [0, 1]
        assert [(m.shot_x, m.shot_y) for m in got] == [(10.0, 20.0), (30.0, 40.0)]
        assert (got[0].x, got[0].y) == pytest.approx((20.0, 40.0))

    def test_flags_a_dot_outside_the_original_instead_of_dropping_it(self):
        """Window chrome and desktop have no counterpart in the original.

        Dropping them would quietly shrink the export; flagging lets it report how
        many were lost and why.
        """
        dots = [_dot(100.0, 100.0), _dot(5000.0, 100.0), _dot(100.0, -50.0)]

        got = map_dots(dots, _result(), (2000, 3000, 3))

        assert len(got) == 3
        assert [m.in_bounds for m in got] == [True, False, False]

    def test_no_dots_maps_to_no_rows(self):
        assert map_dots([], _result(), (2000, 3000, 3)) == []


# ─────────────────────────────────────────────────
# AGAINST A REAL ALIGNMENT
# ─────────────────────────────────────────────────

class TestAgainstAlign:
    def test_round_trip_through_a_measured_transform(self):
        """A screenshot point maps back to where it was taken from.

        The screenshot is built from the original at a known scale and offset, so
        the answer is known independently of `align`: `(p - offset) / scale`.
        """
        original = _textured_original(seed=5)
        scale, dx, dy = 0.5, 12, 20
        shot = _screenshot_from(original, scale=scale, dx=dx, dy=dy)
        res = align(shot, original)
        assert res.ok, res.reason

        got = to_original([(400.0, 300.0)], res)[0]

        assert got == pytest.approx(((400.0 - dx) / scale, (300.0 - dy) / scale),
                                    abs=2.0)

    def test_agrees_with_warp_to_screenshot(self):
        """`warp_to_screenshot` inverts the same transform, so the two must agree."""
        original = _textured_original(seed=11)
        shot = _screenshot_from(original, scale=0.5, dx=12, dy=20)
        res = align(shot, original)
        assert res.ok, res.reason

        pt = np.float64([[[640.0, 480.0]]])
        # Forward through the module, then back the way warping does it.
        forward = to_original([(640.0, 480.0)], res)[0]
        back = cv2.perspectiveTransform(
            np.float64([[[forward[0] * res.scale, forward[1] * res.scale]]]),
            np.linalg.inv(res.H))[0][0]

        assert back == pytest.approx(pt[0][0], abs=1e-6)
