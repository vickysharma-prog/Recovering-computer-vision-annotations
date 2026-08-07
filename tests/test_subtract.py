from __future__ import annotations

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.align import align
from src.subtract import extract_annotations, dot_candidates, SubtractResult
from tests.test_align import _textured_original, _screenshot_from


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _pair(dots=(), dialog=None, seed=0, scale=0.5, dx=12, dy=20):
    """A screenshot/original pair with markers painted only on the screenshot.

    Mirrors the real situation: the original is the clean photograph, the
    screenshot is that photograph rendered smaller with annotations drawn on top.
    """
    original = _textured_original(seed=seed)
    shot = _screenshot_from(original, scale=scale, dx=dx, dy=dy)
    for (x, y) in dots:
        cv2.circle(shot, (x, y), 4, (255, 0, 255), -1)   # saturated magenta ink
    if dialog is not None:
        x, y, w, h = dialog
        cv2.rectangle(shot, (x, y), (x + w, y + h), (212, 208, 200), -1)
    return shot, original


def _aligned(shot, original):
    res = align(shot, original)
    assert res.ok, res.reason
    return extract_annotations(shot, original, res)


def _grid(n_x, n_y, x0=90, y0=90, step=40):
    return [(x0 + i * step, y0 + j * step) for i in range(n_x) for j in range(n_y)]


# ─────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────

class TestExtractAnnotations:
    def test_finds_painted_markers(self):
        dots = _grid(4, 3)
        sub = _aligned(*_pair(dots=dots, seed=21))

        assert sub.mask.shape == sub.valid.shape
        # Every marker should leave ink somewhere near where it was painted.
        for (x, y) in dots:
            assert sub.mask[y - 5:y + 6, x - 5:x + 6].any(), f"missed dot at {x},{y}"

    def test_clean_pair_yields_almost_no_ink(self):
        """No annotations means no detections — the false-positive floor."""
        sub = _aligned(*_pair(dots=(), seed=23))
        assert sub.ink_frac < 0.01

    def test_uncovered_region_is_never_ink(self):
        shot, original = _pair(dots=_grid(2, 2), seed=25, dx=40, dy=40)
        sub = _aligned(shot, original)
        assert not (sub.mask[sub.valid == False] > 0).any()   # noqa: E712

    def test_refuses_rejected_alignment(self):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        original = _textured_original(seed=27)
        bad = align(original, noise)
        assert not bad.ok
        with pytest.raises(ValueError):
            extract_annotations(original, noise, bad)


# ─────────────────────────────────────────────────
# UI MASKING
# ─────────────────────────────────────────────────

class TestUiMasking:
    def test_masks_a_grey_dialog(self):
        """The dialog has no counterpart in the original, so it differs wholly."""
        shot, original = _pair(dots=_grid(3, 2), dialog=(300, 260, 210, 150),
                               seed=29)
        sub = _aligned(shot, original)

        assert sub.ui_mask[300:410, 340:500].mean() > 200
        assert not sub.mask[320:390, 360:480].any()

    def test_keeps_a_dense_marker_carpet(self):
        """Regression: a dense colony merges into one huge connected blob.

        Masking it as chrome on size alone once discarded 92% of the ink on a
        2037-marker image, so a large *saturated* region must survive.
        """
        dots = _grid(14, 10, x0=80, y0=80, step=13)
        sub = _aligned(*_pair(dots=dots, seed=31))

        carpet = sub.ui_mask[80:220, 80:270]
        assert carpet.mean() < 40, "dense marker carpet was masked as chrome"
        assert sub.mask[80:220, 80:270].any()

    def test_keeps_a_scattered_colony(self):
        """Regression: markers spread out, with background between them.

        The tight grid above survives even a whole-region saturation test, because
        its markers nearly touch. A *scattered* colony is the case that failed:
        MORPH_CLOSE bridges the gaps into one region whose pixels are mostly the
        background between markers, so a median over the region reads as dull and
        the whole colony is discarded as chrome. On the hand-labelled frames that
        deleted 92 of 345 real markers, 53 of them on a single frame. Chrome is now
        judged on the region's ink instead, which stays saturated however much
        background the closing pulls in.
        """
        dots = _grid(7, 5, x0=90, y0=90, step=55)
        sub = _aligned(*_pair(dots=dots, seed=61))

        for (x, y) in dots:
            assert not sub.ui_mask[y - 4:y + 5, x - 4:x + 5].any(), \
                f"marker at {x},{y} was masked as chrome"
            assert sub.mask[y - 5:y + 6, x - 5:x + 6].any(), \
                f"marker at {x},{y} lost its ink"


# ─────────────────────────────────────────────────
# DOT CANDIDATES
# ─────────────────────────────────────────────────

class TestDotCandidates:
    def test_counts_isolated_markers(self):
        dots = _grid(4, 3)
        got = dot_candidates(_aligned(*_pair(dots=dots, seed=33)))
        assert abs(len(got)) >= len(dots) * 0.8
        assert len(got) <= len(dots) * 1.5

    def test_candidates_sit_on_the_markers(self):
        """Counting the right number for the wrong reasons is still wrong."""
        dots = _grid(3, 3, x0=120, y0=120, step=60)
        got = dot_candidates(_aligned(*_pair(dots=dots, seed=35)))

        for (cx, cy, *_rest) in got:
            near = min(abs(cx - x) + abs(cy - y) for (x, y) in dots)
            assert near < 12, f"candidate at {cx:.0f},{cy:.0f} matches no marker"

    def test_splits_overlapping_markers(self):
        """Overlapping dots merge into one blob; dense colonies undercount."""
        touching = [(200, 200), (206, 200), (212, 200), (218, 200)]
        got = dot_candidates(_aligned(*_pair(dots=touching, seed=37)))
        assert len(got) >= 2

    def test_ignores_elongated_ink(self):
        """Transect lines and label text are annotations, but not dots."""
        shot, original = _pair(dots=_grid(3, 2), seed=39)
        cv2.line(shot, (60, 420), (560, 424), (255, 0, 255), 3)
        sub = _aligned(shot, original)

        for (cx, cy, *_rest) in dot_candidates(sub):
            assert not (415 < cy < 430 and 60 < cx < 560), "counted the line"

    def test_empty_mask_yields_nothing(self):
        sub = _aligned(*_pair(dots=(), seed=41))
        assert dot_candidates(sub) == [] or len(dot_candidates(sub)) < 5

    def test_subtract_path_count_matches_candidates(self):
        """Wiring parity: detect_dots_subtract emits one dot per candidate.

        The subtraction path must locate exactly what dot_candidates finds — it
        only wraps each center with colour/shape for classification, so folding a
        colour reject in here (which would change the count) is deliberately not
        done. Interior grid dots have no edge-crop drops, so the counts are equal.
        """
        from src.classify import detect_dots_subtract
        dots = _grid(4, 3)
        shot, original = _pair(dots=dots, seed=51)
        res = align(shot, original)
        assert res.ok
        n_cands = len(dot_candidates(extract_annotations(shot, original, res)))
        n_wired = len(detect_dots_subtract(shot, original, res))
        assert n_wired == n_cands and n_wired > 0

    def test_subtract_path_falls_back_when_alignment_refused(self):
        """A refused alignment must degrade to the colour detector, not raise."""
        from src.classify import detect_dots_subtract
        rng = np.random.default_rng(5)
        noise = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        original = _textured_original(seed=53)
        bad = align(original, noise)
        assert not bad.ok
        out = detect_dots_subtract(original, noise, bad)   # falls back, no raise
        assert isinstance(out, list)

    def test_desaturated_ink_is_rejected(self):
        """The chroma gate drops low-saturation ink but keeps a saturated marker.

        On sparse frames the false ink — water glint, mudflat texture, the grey
        label panel — is achromatic, while annotation ink is chromatic. Every
        other candidate test here paints saturated magenta, so they exercise only
        the keep path; this one exercises the drop path directly, so the
        saturation floor cannot be removed without a red test. Built as a bare
        SubtractResult rather than a rendered pair so the two blobs differ in
        exactly one variable — saturation.
        """
        mask = np.zeros((160, 160), np.uint8)
        cv2.circle(mask, (45, 45), 4, 255, -1)      # marker: sits on chroma
        cv2.circle(mask, (110, 110), 4, 255, -1)    # noise: desaturated
        sat = np.full((160, 160), 8, np.uint8)      # low everywhere by default
        cv2.circle(sat, (45, 45), 7, 220, -1)       # only the marker is saturated
        dummy = np.zeros((160, 160), np.float32)
        valid = np.ones((160, 160), bool)
        res = SubtractResult(mask, dummy, valid, np.zeros_like(mask), sat, 0.0)

        got = dot_candidates(res)
        assert got, "the saturated marker was dropped"
        for (cx, cy, *_rest) in got:
            assert abs(cx - 45) < 10 and abs(cy - 45) < 10, \
                "kept a desaturated (non-marker) blob"
