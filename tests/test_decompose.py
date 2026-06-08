"""
Unit tests for src/decompose.py

Focused on pipeline behaviour.
"""

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
def format_a() -> np.ndarray:
    """FORMAT_A: left 55% colorful aerial, right 45% grey dialog."""
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
def format_b() -> np.ndarray:
    """FORMAT_B: full-frame colorful aerial, no dialog."""
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :, 0] = 60
    img[:, :, 1] = 120
    img[:, :, 2] = 180
    return img


@pytest.fixture(scope="module")
def format_a_with_bars() -> np.ndarray:
    """FORMAT_A with title bar + taskbar (grey strips top/bottom)."""
    img = np.zeros((660, 900, 3), dtype=np.uint8)
    img[:30, :, :] = 200
    img[30:630, :495, 0] = 60
    img[30:630, :495, 1] = 120
    img[30:630, :495, 2] = 180
    img[30:630, 495:, :] = 210
    img[630:, :, :] = 200
    return img


@pytest.fixture(scope="module")
def result_a() -> DecompositionResult:
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
def result_b() -> DecompositionResult:
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
# _is_grey - only the bug-catch test matters
# ─────────────────────────────────────────────────

class TestIsGrey:

    def test_no_uint8_overflow(self):
        """
        uint8: |R=10 - G=250| wraps to 16 (WRONG → falsely grey).
        int16: |10 - 250| = 240 (CORRECT → not grey).
        Catches real bug found during prototype.
        """
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 250
        img[:, :, 2] = 200
        assert not _is_grey(img).any()

    def test_grey_pixels_detected(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        assert _is_grey(img).all()

    def test_colorful_pixels_rejected(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 2] = 255
        assert not _is_grey(img).any()


# ─────────────────────────────────────────────────
# _validate_image
# ─────────────────────────────────────────────────

class TestValidateImage:

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            _validate_image(None, "fn")

    def test_2d_raises_value_error(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            _validate_image(img, "fn")

    def test_empty_raises_value_error(self):
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _validate_image(img, "fn")


# ─────────────────────────────────────────────────
# detect_format
# ─────────────────────────────────────────────────

class TestDetectFormat:

    def test_format_a_detected(self, decomposer, format_a):
        fmt, conf = decomposer.detect_format(format_a)
        assert fmt == 'FORMAT_A'
        assert conf > 0.5

    def test_format_b_detected(self, decomposer, format_b):
        fmt, _ = decomposer.detect_format(format_b)
        assert fmt == 'FORMAT_B'

    def test_confidence_in_range(self, decomposer, format_a):
        _, conf = decomposer.detect_format(format_a)
        assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────
# find_dialog_boundary
# ─────────────────────────────────────────────────

class TestFindDialogBoundary:

    def test_boundary_in_valid_range(self, decomposer, format_a):
        w = format_a.shape[1]
        bx, _ = decomposer.find_dialog_boundary(format_a)
        assert w * 0.35 <= bx <= w * 0.70

    def test_dialog_minimum_80px(self, decomposer, format_a):
        w = format_a.shape[1]
        bx, _ = decomposer.find_dialog_boundary(format_a)
        assert (w - bx) >= 80

    def test_confidence_in_range(self, decomposer, format_a):
        _, conf = decomposer.find_dialog_boundary(format_a)
        assert 0.3 <= conf <= 1.0


# ─────────────────────────────────────────────────
# detect_bars
# ─────────────────────────────────────────────────

class TestDetectBars:

    def test_no_bars_in_colorful_image(
        self, decomposer, format_b
    ):
        title, taskbar = decomposer.detect_bars(format_b)
        assert title is None
        assert taskbar is None

    def test_bars_detected(
        self, decomposer, format_a_with_bars
    ):
        title, taskbar = decomposer.detect_bars(
            format_a_with_bars
        )
        assert title is not None
        assert taskbar is not None


# ─────────────────────────────────────────────────
# decompose - end to end
# ─────────────────────────────────────────────────

class TestDecompose:

    def test_format_a_has_dialog(self, result_a):
        assert result_a.has_dialog()
        assert result_a.dialog is not None

    def test_format_b_no_dialog(self, result_b):
        assert not result_b.has_dialog()
        assert result_b.dialog is None

    def test_no_pixels_lost(self, result_a):
        """aerial + dialog = full width. No pixels lost."""
        total = (
            result_a.aerial_width()
            + result_a.dialog.shape[1]
        )
        assert total == 900

    def test_aerial_dialog_same_height(self, result_a):
        assert (
            result_a.aerial.shape[0]
            == result_a.dialog.shape[0]
        )

    def test_bars_excluded_from_content(
        self, decomposer, format_a_with_bars
    ):
        result = decomposer.decompose(format_a_with_bars)
        assert result.aerial_height() < 660

    def test_stateless(self, decomposer, format_a):
        """Same result on repeated calls."""
        r1 = decomposer.decompose(format_a)
        r2 = decomposer.decompose(format_a)
        assert r1.format == r2.format
        assert r1.boundary_x == r2.boundary_x

    def test_invalid_input_raises(self, decomposer):
        with pytest.raises(TypeError):
            decomposer.decompose(None)

    def test_bars_correct_content_height(
        self, decomposer, format_a_with_bars
    ):
        result = decomposer.decompose(format_a_with_bars)
        h = format_a_with_bars.shape[0]
        title_h = (
            result.title_bar.height
            if result.title_bar else 0
        )
        taskbar_h = (
            result.taskbar.height
            if result.taskbar else 0
        )
        assert result.aerial_height() == h - title_h - taskbar_h


# ─────────────────────────────────────────────────
# BarInfo
# ─────────────────────────────────────────────────

class TestBarInfo:

    def test_immutable(self):
        bar = BarInfo(y=0, height=30)
        with pytest.raises(dataclasses.FrozenInstanceError):
            bar.y = 10  # type: ignore

    def test_equality(self):
        assert BarInfo(y=0, height=30) == BarInfo(y=0, height=30)


# ─────────────────────────────────────────────────
# Real Image
# ─────────────────────────────────────────────────

class TestRealImage:
    """Integration tests on real dataset screenshot."""

    def test_no_crash(self, decomposer, real_screenshot):
        result = decomposer.decompose(real_screenshot)
        assert result is not None

    def test_valid_format(self, decomposer, real_screenshot):
        result = decomposer.decompose(real_screenshot)
        assert result.format in ('FORMAT_A', 'FORMAT_B')

    def test_aerial_non_empty(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert result.aerial.size > 0

    def test_stateless_on_real(
        self, decomposer, real_screenshot
    ):
        r1 = decomposer.decompose(real_screenshot)
        r2 = decomposer.decompose(real_screenshot)
        assert r1.format == r2.format
        assert r1.boundary_x == r2.boundary_x