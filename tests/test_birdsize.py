from __future__ import annotations

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.birdsize import (MAX_PX, MIN_DOTS, SizeEstimate, box_from_size,
                          frame_bird_size)


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _scene(bird_px: int, n: int = 20, spacing: int = 120, ground=(90, 120, 80),
           bird=(230, 230, 235), size=(900, 900), seed: int = 0):
    """Textured ground with `n` discs of a known diameter, and their centres.

    The ground is noisy on purpose: a measurement that only works on a flat
    background would pass here and fail on marsh.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = ground
    img = np.clip(img.astype(np.int16)
                  + rng.integers(-14, 15, (h, w, 3)), 0, 255).astype(np.uint8)

    pts, k = [], 0
    for y in range(spacing, h - spacing, spacing):
        for x in range(spacing, w - spacing, spacing):
            if k >= n:
                break
            cv2.circle(img, (x, y), bird_px // 2, bird, -1)
            pts.append((float(x), float(y)))
            k += 1
    return img, pts


# ─────────────────────────────────────────────────
# MEASUREMENT
# ─────────────────────────────────────────────────

class TestFrameBirdSize:
    def test_recovers_a_known_size(self):
        img, pts = _scene(bird_px=14)

        est = frame_bird_size(img, pts)

        assert est.ok
        assert est.median_px == pytest.approx(14, abs=3)
        assert est.n_measured >= MIN_DOTS

    def test_separates_two_frames_of_different_scale(self):
        """The whole point: the same species is smaller on a coarser photograph."""
        small = frame_bird_size(*_scene(bird_px=10, seed=1))
        large = frame_bird_size(*_scene(bird_px=24, seed=2))

        assert small.ok and large.ok
        assert large.median_px > small.median_px * 1.6

    def test_finds_a_dark_bird_on_a_pale_ground(self):
        """Contrast against local background, not an absolute colour: birds sit on
        sand as often as on marsh, and there they are darker, not paler."""
        img, pts = _scene(bird_px=14, ground=(215, 210, 200), bird=(40, 40, 45))

        est = frame_bird_size(img, pts)

        assert est.ok
        assert est.median_px == pytest.approx(14, abs=3)

    def test_too_few_dots_measures_nothing(self):
        img, pts = _scene(bird_px=14, n=3)

        est = frame_bird_size(img, pts)

        assert not est.ok
        assert est.median_px is None

    def test_no_dots_at_all(self):
        img, _ = _scene(bird_px=14)

        assert not frame_bird_size(img, []).ok

    def test_dots_at_the_edge_are_skipped_not_crashed(self):
        img, pts = _scene(bird_px=14)
        edge = [(1.0, 1.0), (float(img.shape[1] - 2), float(img.shape[0] - 2))]

        est = frame_bird_size(img, edge + pts)

        assert est.ok                      # the interior dots still carry it

    def test_a_merged_pair_does_not_drag_the_median(self):
        """Touching birds measure as one large object. The median has to absorb it."""
        img, pts = _scene(bird_px=12)
        for x, y in pts[:4]:               # a second disc against the first
            cv2.circle(img, (int(x) + 7, int(y)), 6, (230, 230, 235), -1)

        est = frame_bird_size(img, pts)

        assert est.ok
        assert est.median_px == pytest.approx(12, abs=4)


# ─────────────────────────────────────────────────
# BOX
# ─────────────────────────────────────────────────

class TestBoxFromSize:
    def test_leaves_a_margin_around_the_bird(self):
        assert box_from_size(20.0) > 20

    def test_scales_with_the_bird(self):
        assert box_from_size(24.0) > box_from_size(10.0)

    def test_guards_clamp_a_failed_measurement(self):
        assert box_from_size(0.5) == 16
        assert box_from_size(500.0) == 120

    def test_the_upper_guard_never_binds_on_measured_sizes(self):
        """Nothing measured on the 25 frames comes near 120px."""
        for px in (7.0, 12.0, 21.0):
            assert box_from_size(px) < 120

    def test_the_lower_guard_binds_on_the_coarsest_frames(self):
        """At 3.3cm/px a gull is about 12px long, and 1.3x that is under the floor.

        Worth stating rather than hiding: on the coarsest frames the box is the
        floor rather than a measurement. It barely matters there — 1.3 x 11.7 is
        15.2 against a floor of 16 — but it means those frames are not carrying a
        measured box, and `box_measured` in the export is what says so.
        """
        assert box_from_size(11.7) == 16
        assert box_from_size(20.0) > 16


class TestSizeEstimate:
    def test_ok_is_false_without_a_median(self):
        assert not SizeEstimate(None, None, None, 0, 0, False).ok

    def test_max_px_rejects_a_whole_patch(self):
        """A component filling the patch is map ink or merged birds, not a bird."""
        assert MAX_PX < 100
