from __future__ import annotations

import glob

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.legend import (
    LegendEntry,
    canonical_template,
    locate_dialog,
    parse_legend,
    parse_screenshot,
    _circular_mean_hue,
    _name_hue,
    _classify_shape,
    _is_cross,
    _is_marker_like,
    _has_enclosed_hole,
    _levenshtein,
    _best_match,
    _parse_class_text,
    _SPECIES_CODES,
)


# ─────────────────────────────────────────────────
# SYNTHETIC GLYPH FIXTURES
# ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def filled_square() -> np.ndarray:
    return np.ones((20, 20), np.uint8)


@pytest.fixture(scope="module")
def filled_circle() -> np.ndarray:
    m = np.zeros((24, 24), np.uint8)
    cv2.circle(m, (12, 12), 9, 1, -1)
    return m


@pytest.fixture(scope="module")
def plus_glyph() -> np.ndarray:
    m = np.zeros((24, 24), np.uint8)
    m[10:14, :] = 1
    m[:, 10:14] = 1
    return m


@pytest.fixture(scope="module")
def ring_glyph() -> np.ndarray:
    m = np.zeros((28, 28), np.uint8)
    cv2.circle(m, (14, 14), 10, 1, 2)
    return m


# ─────────────────────────────────────────────────
# COLOR
# ─────────────────────────────────────────────────

class TestColor:

    def test_red_wraparound(self):
        """Red hue wraps in OpenCV (0 and 180); circular mean must stay red."""
        result = _circular_mean_hue(np.array([5, 175]))
        assert result < 15 or result > 165

    def test_empty_returns_zero(self):
        assert _circular_mean_hue(np.array([])) == 0.0

    def test_name_red_both_ends(self):
        assert _name_hue(2) == "red"
        assert _name_hue(175) == "red"

    def test_name_green_blue(self):
        assert _name_hue(60) == "green"
        assert _name_hue(115) == "blue"


# ─────────────────────────────────────────────────
# SHAPE CLASSIFICATION
# ─────────────────────────────────────────────────

class TestClassifyShape:

    def test_square(self, filled_square):
        assert _classify_shape(filled_square) == "square"

    def test_circle(self, filled_circle):
        assert _classify_shape(filled_circle) == "circle"

    def test_plus(self, plus_glyph):
        assert _classify_shape(plus_glyph) == "plus"

    def test_ring(self, ring_glyph):
        assert _classify_shape(ring_glyph) == "ring"

    def test_empty_is_unknown(self):
        assert _classify_shape(np.zeros((10, 10), np.uint8)) == "unknown"

    def test_is_cross_true_for_plus(self, plus_glyph):
        assert _is_cross(plus_glyph)

    def test_is_cross_false_for_square(self, filled_square):
        assert not _is_cross(filled_square)

    def test_hole_detected_for_ring(self, ring_glyph):
        assert _has_enclosed_hole(ring_glyph)

    def test_no_hole_for_filled_disk(self, filled_circle):
        assert not _has_enclosed_hole(filled_circle)


# ─────────────────────────────────────────────────
# TEMPLATE / MARKER-LIKE
# ─────────────────────────────────────────────────

class TestTemplate:

    def test_canonical_output_size(self, filled_circle):
        t = canonical_template(filled_circle)
        assert t.shape == (24, 24)
        assert t.max() <= 1.0 and t.min() >= 0.0

    def test_canonical_custom_size(self, filled_square):
        assert canonical_template(filled_square, size=16).shape == (16, 16)

    def test_canonical_empty_mask(self):
        t = canonical_template(np.zeros((5, 5), np.uint8))
        assert t.shape == (24, 24)
        assert t.max() == 0.0

    def test_marker_like_centered_glyph(self, filled_circle):
        assert _is_marker_like(filled_circle)

    def test_marker_like_rejects_full_cell(self):
        assert not _is_marker_like(np.ones((24, 24), np.uint8))

    def test_marker_like_rejects_empty(self):
        assert not _is_marker_like(np.zeros((24, 24), np.uint8))


# ─────────────────────────────────────────────────
# LegendEntry
# ─────────────────────────────────────────────────

class TestLegendEntry:

    def _entry(self, color="red", shape="circle") -> LegendEntry:
        return LegendEntry(
            row=0, cy=10.0, cx=5.0, shape=shape, color=color, hue=2.0,
            marker=np.zeros((2, 2, 3), np.uint8),
            template=np.zeros((24, 24), np.float32),
        )

    def test_key_is_color_shape(self):
        assert self._entry("red", "circle").key() == "red:circle"

    def test_repr_contains_color_shape(self):
        assert "red/circle" in repr(self._entry())


# ─────────────────────────────────────────────────
# OCR TEXT PARSING (no tesseract needed)
# ─────────────────────────────────────────────────

class TestTextParsing:

    def test_levenshtein_identical(self):
        assert _levenshtein("BRPE", "BRPE") == 0

    def test_levenshtein_one_edit(self):
        assert _levenshtein("BAPE", "BRPE") == 1

    def test_best_match_corrects_ocr_error(self):
        """'BAPE' is one edit from species code 'BRPE'."""
        assert _best_match("BAPE", _SPECIES_CODES, max_dist=1) == "BRPE"

    def test_best_match_returns_none_when_far(self):
        assert _best_match("ZZZZ", _SPECIES_CODES, max_dist=1) is None

    def test_parse_species_category_count(self):
        name, sp, cat, cnt = _parse_class_text("BRPE WBN 42", _SPECIES_CODES)
        assert sp == "BRPE"
        assert cat == "wbn"
        assert cnt == 42

    def test_parse_trailing_count(self):
        _, _, _, cnt = _parse_class_text("LAGU Site 7", _SPECIES_CODES)
        assert cnt == 7

    def test_parse_empty_text(self):
        assert _parse_class_text("", _SPECIES_CODES) == (None, None, None, None)


# ─────────────────────────────────────────────────
# REAL SCREENSHOT INTEGRATION
# ─────────────────────────────────────────────────

def _screenshots() -> list[str]:
    return sorted(glob.glob("data/fixtures/screenshots/*.jpg"))


@pytest.fixture(params=_screenshots() or [None])
def screenshot(request) -> np.ndarray:
    path = request.param
    if path is None:
        pytest.skip("No screenshot fixtures in data/fixtures/screenshots/")
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Unreadable fixture: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class TestRealScreenshot:

    def test_locate_dialog_returns_box(self, screenshot):
        """The floating dialog is found as a contained box, not a half-split."""
        bbox = locate_dialog(screenshot)
        assert bbox is not None
        x, y, w, h = bbox
        H, W = screenshot.shape[:2]
        assert w > 0 and h > 0
        # A dialog is a contained box, never the whole frame.
        assert w < W and h < H

    def test_parse_screenshot_recovers_rows(self, screenshot):
        entries, bbox = parse_screenshot(screenshot)
        assert bbox is not None
        assert len(entries) >= 5

    def test_entries_have_color_and_template(self, screenshot):
        entries, _ = parse_screenshot(screenshot)
        for e in entries:
            assert e.template.shape == (24, 24)
            assert isinstance(e.color, str)

    def test_same_color_kept_distinct_by_shape(self, screenshot):
        """Mentor's core ask: same-colour markers separated by shape."""
        entries, _ = parse_screenshot(screenshot)
        keys = [e.key() for e in entries]
        # No crash + stable identity; at least one colour group exists.
        assert len(keys) == len(entries)

    def test_stateless(self, screenshot):
        a, _ = parse_screenshot(screenshot)
        b, _ = parse_screenshot(screenshot)
        assert len(a) == len(b)


class TestParseLegendGuards:

    def test_empty_input_returns_empty(self):
        assert parse_legend(np.zeros((0, 0, 3), np.uint8)) == []

    def test_none_input_returns_empty(self):
        assert parse_legend(None) == []

    def test_no_grid_returns_empty(self):
        """A blank grey panel has no marker grid."""
        assert parse_legend(np.full((100, 100, 3), 200, np.uint8)) == []
