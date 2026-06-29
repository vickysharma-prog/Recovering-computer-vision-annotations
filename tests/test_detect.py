from __future__ import annotations

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.detect import (
    DetectedDot,
    DetectionResult,
    DotDetector,
    circular_mean_hue,
    _validate_aerial,
    _hsv_mask,
    _vegetation_boost,
    _match_colors_to_species,
    _select_by_count,
)


# ─────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector() -> DotDetector:
    return DotDetector()


@pytest.fixture(scope="module")
def aerial_with_red_dot() -> np.ndarray:
    """Green aerial with one bright red dot at center."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :, 1] = 80
    cv2.circle(img, (200, 150), 6, (255, 30, 30), -1)
    return img


@pytest.fixture(scope="module")
def aerial_no_dots() -> np.ndarray:
    """Uniform grey aerial, no annotation dots."""
    return np.full((300, 400, 3), 180, dtype=np.uint8)

@pytest.fixture(scope="module")
def heavy_vegetation() -> np.ndarray:
    """Aerial dominated by green vegetation (>40% coverage)."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :, 0] = 60   # R
    img[:, :, 1] = 120  # G
    img[:, :, 2] = 60   # B
    return img


@pytest.fixture
def real_aerial() -> np.ndarray:
    path = "data/fixtures/sample_screenshot.png"
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w = img_rgb.shape[1]
    return img_rgb[:, : int(w * 0.55), :]


# ─────────────────────────────────────────────────
# circular_mean_hue
# ─────────────────────────────────────────────────

class TestCircularMeanHue:

    def test_red_wraparound(self):
        """
        Catches real prototype bug: mean([5, 175]) = 90 (wrong).
        Circular mean gives ~0 (correct) for red hue wraparound.
        """
        result = circular_mean_hue(np.array([5, 175]))
        assert result < 15 or result > 165

    def test_empty_returns_zero(self):
        assert circular_mean_hue(np.array([])) == 0.0


# ─────────────────────────────────────────────────
# _validate_aerial
# ─────────────────────────────────────────────────

class TestValidateAerial:

    def test_none_raises(self):
        with pytest.raises(TypeError):
            _validate_aerial(None, "fn")

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            _validate_aerial(
                np.zeros((100, 200), dtype=np.uint8), "fn"
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _validate_aerial(
                np.zeros((0, 0, 3), dtype=np.uint8), "fn"
            )


# ─────────────────────────────────────────────────
# _hsv_mask
# ─────────────────────────────────────────────────

class TestHsvMask:

    def test_red_dot_produces_mask(self, aerial_with_red_dot):
        """Red annotation dot must appear in red HSV mask."""
        bgr = cv2.cvtColor(aerial_with_red_dot, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = _hsv_mask(hsv, [[0, 20], [160, 180]], 80, 70)
        assert mask.any()

    def test_uniform_grey_produces_no_mask(self):
        """Grey background must not trigger annotation detection."""
        img = np.full((50, 50, 3), 180, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = _hsv_mask(hsv, [[0, 20], [160, 180]], 80, 70)
        assert not mask.any()


# ─────────────────────────────────────────────────
# _vegetation_boost
# ─────────────────────────────────────────────────

class TestVegetationBoost:

    def test_no_vegetation_no_boost(self):
        assert _vegetation_boost(0.0) == 0

    def test_low_vegetation_boost(self):
        assert _vegetation_boost(20.0) == 30

    def test_mid_vegetation_boost(self):
        assert _vegetation_boost(30.0) == 50

    def test_high_vegetation_boost(self):
        assert _vegetation_boost(50.0) == 70


# ─────────────────────────────────────────────────
# _match_colors_to_species
# ─────────────────────────────────────────────────

class TestMatchColorsToSpecies:

    def test_most_abundant_gets_largest_color(self):
        """Rank-order: most birds -> largest dot group."""
        mapping = _match_colors_to_species(
            {"red": 90, "yellow": 40},
            {"LAGU": 85, "BRPE": 42},
        )
        assert mapping["LAGU"] == "red"
        assert mapping["BRPE"] == "yellow"

    def test_no_double_assignment(self):
        """Each color assigned to at most one species."""
        mapping = _match_colors_to_species(
            {"red": 50, "yellow": 30, "blue": 20},
            {"LAGU": 48, "BRPE": 28, "ROYT": 18},
        )
        assigned = list(mapping.values())
        assert len(assigned) == len(set(assigned))

    def test_empty_inputs_return_empty(self):
        assert _match_colors_to_species({}, {"LAGU": 50}) == {}
        assert _match_colors_to_species({"red": 50}, {}) == {}


# ─────────────────────────────────────────────────
# _select_by_count
# ─────────────────────────────────────────────────

class TestSelectByCount:

    def _make_dots(self, color: str, n: int) -> list[DetectedDot]:
        return [
            DetectedDot(
                cx=float(i * 10), cy=10.0,
                color=color, species=None,
                score=float(n - i),
                area=50, circularity=0.8,
            )
            for i in range(n)
        ]

    def test_top_n_selected(self):
        selected, _, _ = _select_by_count(
            {"red": self._make_dots("red", 20)},
            {"LAGU": "red"},
            {"LAGU": 10},
        )
        assert len(selected) == 10

    def test_highest_score_dots_selected(self):
        """Top-N must pick highest scoring dots."""
        selected, _, _ = _select_by_count(
            {"red": self._make_dots("red", 10)},
            {"LAGU": "red"},
            {"LAGU": 3},
        )
        scores = [d.score for d in selected]
        assert scores == sorted(scores, reverse=True)

    def test_species_written_to_selected_dots(self):
        selected, _, _ = _select_by_count(
            {"red": self._make_dots("red", 5)},
            {"LAGU": "red"},
            {"LAGU": 5},
        )
        assert all(d.species == "LAGU" for d in selected)

    def test_unmatched_species_zero_detected(self):
        _, per_species, _ = _select_by_count(
            {"red": self._make_dots("red", 10)},
            {},
            {"LAGU": 10},
        )
        assert per_species["LAGU"]["detected"] == 0

    def test_category_counts_handled_as_metadata(self):
        """category_counts are stored in per_category metadata but not assigned to dots."""
        selected, _, per_cat = _select_by_count(
            {"red": self._make_dots("red", 100)},
            {"BRPE": "red"},
            {"BRPE": 50},
            {"BRPE_WBN": 30, "BRPE_OtherAdultsInColony": 20},
        )
        assert all(d.category is None for d in selected)
        assert "BRPE_WBN" in per_cat
        assert "BRPE_OtherAdultsInColony" in per_cat
        assert per_cat["BRPE_WBN"]["assigned"] == 0


# ─────────────────────────────────────────────────
# DotDetector end-to-end
# ─────────────────────────────────────────────────

class TestDotDetector:

    def test_invalid_input_raises(self, detector):
        with pytest.raises(TypeError):
            detector.detect(None, {"LAGU": 10})

    def test_zero_birds_skipped(
        self, detector, aerial_with_red_dot
    ):
        result = detector.detect(aerial_with_red_dot, {"LAGU": 0})
        assert result.status == "zero_birds"
        assert result.total_detected == 0

    def test_none_csv_skipped(
        self, detector, aerial_with_red_dot
    ):
        result = detector.detect(aerial_with_red_dot, None)
        assert result.status == "zero_birds"

    def test_red_dot_detected(
        self, detector, aerial_with_red_dot
    ):
        result = detector.detect(aerial_with_red_dot, {"LAGU": 1})
        assert result.total_detected >= 1

    def test_no_dots_aerial_detects_nothing(
        self, detector, aerial_no_dots
    ):
        result = detector.detect(aerial_no_dots, {"LAGU": 10})
        assert result.total_detected == 0

    def test_high_vegetation_triggers_boost(
        self, detector, heavy_vegetation
    ):
        result = detector.detect(heavy_vegetation, {"LAGU": 1})
        assert result.green_s_boost > 0

    def test_detected_count_matches_dots_tuple(
        self, detector, aerial_with_red_dot
    ):
        """total_detected must equal len(dots)."""
        result = detector.detect(aerial_with_red_dot, {"LAGU": 1})
        assert result.total_detected == len(result.dots)

    def test_stateless(self, detector, aerial_with_red_dot):
        """Same result on repeated calls - no internal state."""
        r1 = detector.detect(aerial_with_red_dot, {"LAGU": 1})
        r2 = detector.detect(aerial_with_red_dot, {"LAGU": 1})
        assert r1.total_detected == r2.total_detected
        assert r1.status == r2.status


# ─────────────────────────────────────────────────
# Real image integration
# ─────────────────────────────────────────────────

class TestRealImage:

    def test_no_crash(self, detector, real_aerial):
        result = detector.detect(real_aerial, {"LAGU": 50})
        assert result is not None

    def test_valid_result_fields(self, detector, real_aerial):
        result = detector.detect(real_aerial, {"LAGU": 50})
        assert isinstance(result.status, str)
        assert result.total_detected >= 0

    def test_stateless_on_real(self, detector, real_aerial):
        r1 = detector.detect(real_aerial, {"LAGU": 50})
        r2 = detector.detect(real_aerial, {"LAGU": 50})
        assert r1.total_detected == r2.total_detected