
from __future__ import annotations

import dataclasses
import cv2
import numpy as np
import pytest

from src.decompose import (
    BarInfo,
    DecompositionResult,
    ScreenshotDecomposer,
    _is_grey,
    _validate_image,
)


# ─────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def decomposer() -> ScreenshotDecomposer:
    return ScreenshotDecomposer()


@pytest.fixture(scope="module")
def with_dialog() -> np.ndarray:
    """Screenshot with dialog: left 55% aerial, right 45% grey."""
    rng = np.random.default_rng(42)
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    noise = rng.integers(-30, 30, (600, 495, 3))
    aerial = np.clip(
        np.array([60, 120, 180]) + noise, 0, 255
    ).astype(np.uint8)
    img[:, :495, :] = aerial
    img[:, 495:, :] = 210
    return img


@pytest.fixture(scope="module")
def no_dialog() -> np.ndarray:
    """Full-frame aerial screenshot, no dialog."""
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :, 0] = 60
    img[:, :, 1] = 120
    img[:, :, 2] = 180
    return img


@pytest.fixture(scope="module")
def with_bars() -> np.ndarray:
    """Screenshot with title bar + taskbar (30px each)."""
    img = np.zeros((660, 900, 3), dtype=np.uint8)
    img[:30, :, :] = 200
    img[30:630, :495, 0] = 60
    img[30:630, :495, 1] = 120
    img[30:630, :495, 2] = 180
    img[30:630, 495:, :] = 210
    img[630:, :, :] = 200
    return img


@pytest.fixture(scope="module")
def result_with_dialog() -> DecompositionResult:
    d = ScreenshotDecomposer()
    rng = np.random.default_rng(42)
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    noise = rng.integers(-30, 30, (600, 495, 3))
    aerial = np.clip(
        np.array([60, 120, 180]) + noise, 0, 255
    ).astype(np.uint8)
    img[:, :495, :] = aerial
    img[:, 495:, :] = 210
    return d.decompose(img)


@pytest.fixture(scope="module")
def result_no_dialog() -> DecompositionResult:
    d = ScreenshotDecomposer()
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :, 0] = 60
    img[:, :, 1] = 120
    img[:, :, 2] = 180
    return d.decompose(img)


@pytest.fixture
def real_screenshot() -> np.ndarray:
    path = "data/fixtures/sample_screenshot.png"
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Real fixture not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────
# _is_grey
# ─────────────────────────────────────────────────

class TestIsGrey:

    def test_no_uint8_overflow(self):
        """
        Catches real prototype bug:
        uint8 |R=10 - G=250| wraps to 16 → falsely grey.
        int16 |10 - 250| = 240 → correctly not grey.
        """
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 250
        img[:, :, 2] = 200
        assert not _is_grey(img).any()

    def test_grey_detected(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        assert _is_grey(img).all()

    def test_colorful_rejected(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 2] = 255
        assert not _is_grey(img).any()


# ─────────────────────────────────────────────────
# _validate_image
# ─────────────────────────────────────────────────

class TestValidateImage:

    def test_none_raises(self):
        with pytest.raises(TypeError):
            _validate_image(None, "fn")

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            _validate_image(
                np.zeros((100, 200), dtype=np.uint8), "fn"
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _validate_image(
                np.zeros((0, 0, 3), dtype=np.uint8), "fn"
            )


# ─────────────────────────────────────────────────
# Dialog detection
# ─────────────────────────────────────────────────

class TestDialogDetection:

    def test_dialog_present(
        self, decomposer, with_dialog
    ):
        result = decomposer.decompose(with_dialog)
        assert result.has_dialog is True

    def test_no_dialog(
        self, decomposer, no_dialog
    ):
        result = decomposer.decompose(no_dialog)
        assert result.has_dialog is False

    def test_expect_dialog_override_true(
        self, decomposer, no_dialog
    ):
        """expect_dialog=True forces dialog split."""
        result = decomposer.decompose(
            no_dialog, expect_dialog=True
        )
        assert result.has_dialog is True
        assert result.detection_confidence == 1.0

    def test_expect_dialog_override_false(
        self, decomposer, with_dialog
    ):
        """expect_dialog=False forces no-dialog path."""
        result = decomposer.decompose(
            with_dialog, expect_dialog=False
        )
        assert result.has_dialog is False
        assert result.detection_confidence == 1.0

    def test_confidence_in_range(
        self, decomposer, with_dialog
    ):
        result = decomposer.decompose(with_dialog)
        assert 0.0 <= result.detection_confidence <= 1.0


# ─────────────────────────────────────────────────
# Boundary detection
# ─────────────────────────────────────────────────

class TestBoundaryDetection:

    def test_boundary_in_valid_range(
        self, decomposer, with_dialog
    ):
        w = with_dialog.shape[1]
        bx, _, _ = decomposer.find_dialog_boundary(with_dialog)
        assert w * 0.35 <= bx <= w * 0.70

    def test_dialog_minimum_width(
        self, decomposer, with_dialog
    ):
        w = with_dialog.shape[1]
        bx, _, _ = decomposer.find_dialog_boundary(with_dialog)
        assert (w - bx) >= 80

    def test_confidence_in_range(
        self, decomposer, with_dialog
    ):
        _, conf, _ = decomposer.find_dialog_boundary(with_dialog)
        assert 0.3 <= conf <= 1.0

    def test_candidates_returned(
        self, decomposer, with_dialog
    ):
        """Raw grey/edge/variance candidates are exposed."""
        _, _, cands = decomposer.find_dialog_boundary(with_dialog)
        assert len(cands) == 3
        assert all(isinstance(c, int) for c in cands)


# ─────────────────────────────────────────────────
# Bar detection
# ─────────────────────────────────────────────────

class TestBarDetection:

    def test_no_bars_in_aerial(
        self, decomposer, no_dialog
    ):
        title, taskbar = decomposer.detect_bars(no_dialog)
        assert title is None
        assert taskbar is None

    def test_bars_detected(
        self, decomposer, with_bars
    ):
        title, taskbar = decomposer.detect_bars(with_bars)
        assert title is not None
        assert taskbar is not None


# ─────────────────────────────────────────────────
# Decompose end-to-end
# ─────────────────────────────────────────────────

class TestDecompose:

    def test_dialog_present_splits_regions(
        self, result_with_dialog
    ):
        assert result_with_dialog.has_dialog
        assert result_with_dialog.dialog is not None
        assert result_with_dialog.aerial is not None

    def test_no_dialog_full_aerial(
        self, result_no_dialog
    ):
        assert not result_no_dialog.has_dialog
        assert result_no_dialog.dialog is None

    def test_no_pixels_lost(self, result_with_dialog):
        """aerial + dialog = full width. No pixels dropped."""
        total = (
            result_with_dialog.aerial_width()
            + result_with_dialog.dialog.shape[1]
        )
        assert total == 900

    def test_boundary_candidates_present(
        self, result_with_dialog
    ):
        """Dialog result includes raw boundary candidates."""
        assert result_with_dialog.boundary_candidates is not None
        assert len(result_with_dialog.boundary_candidates) == 3

    def test_no_dialog_candidates_none(
        self, result_no_dialog
    ):
        """No-dialog result has no boundary candidates."""
        assert result_no_dialog.boundary_candidates is None

    def test_regions_same_height(
        self, result_with_dialog
    ):
        assert (
            result_with_dialog.aerial.shape[0]
            == result_with_dialog.dialog.shape[0]
        )

    def test_bars_excluded_from_content(
        self, decomposer, with_bars
    ):
        result = decomposer.decompose(with_bars)
        assert result.aerial_height() < 660

    def test_bars_correct_height(
        self, decomposer, with_bars
    ):
        result = decomposer.decompose(with_bars)
        h = with_bars.shape[0]
        title_h = (
            result.title_bar.height
            if result.title_bar else 0
        )
        taskbar_h = (
            result.taskbar.height
            if result.taskbar else 0
        )
        assert result.aerial_height() == h - title_h - taskbar_h

    def test_stateless(self, decomposer, with_dialog):
        """Same result on repeated calls."""
        r1 = decomposer.decompose(with_dialog)
        r2 = decomposer.decompose(with_dialog)
        assert r1.has_dialog == r2.has_dialog
        assert r1.boundary_x == r2.boundary_x

    def test_invalid_input_raises(self, decomposer):
        with pytest.raises(TypeError):
            decomposer.decompose(None)

# ─────────────────────────────────────────────────
# Real image integration
# ─────────────────────────────────────────────────

class TestRealImage:

    def test_no_crash(self, decomposer, real_screenshot):
        result = decomposer.decompose(real_screenshot)
        assert result is not None

    def test_valid_result(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert isinstance(result.has_dialog, bool)
        assert result.aerial.size > 0

    def test_stateless_on_real(
        self, decomposer, real_screenshot
    ):
        r1 = decomposer.decompose(real_screenshot)
        r2 = decomposer.decompose(real_screenshot)
        assert r1.has_dialog == r2.has_dialog
        assert r1.boundary_x == r2.boundary_x