"""
Unit tests for src/decompose.py

Test organization:
    TestIsGrey             - _is_grey() pure function
    TestSmooth1d           - _smooth1d() pure function
    TestKernelSize         - _kernel_size() pure function
    TestGreyProfile        - _grey_profile() pure function
    TestEdgeProfile        - _edge_profile() pure function
    TestVarianceProfile    - _variance_profile() pure function
    TestValidateImage      - _validate_image() pure function
    TestDetectFormat       - detect_format() method
    TestFindDialogBoundary - find_dialog_boundary() method
    TestDetectBars         - detect_bars() method
    TestDecompose          - decompose() end-to-end
    TestDecompositionResult- dataclass helper methods
    TestBarInfo            - BarInfo frozen dataclass
    TestRealImage          - real fixture (skipped if missing)

All fixtures are module-scoped for performance.
result_a / result_b are self-contained (no cross-deps).
real_screenshot is function-scoped (real I/O).
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
    _edge_profile,
    _grey_profile,
    _is_grey,
    _kernel_size,
    _smooth1d,
    _validate_image,
    _variance_profile,
)


# ─────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def decomposer() -> ScreenshotDecomposer:
    """Shared stateless decomposer instance."""
    return ScreenshotDecomposer()


@pytest.fixture(scope="module")
def format_a() -> np.ndarray:
    """
    Synthetic FORMAT_A screenshot.
    Left 55% = colorful aerial (blue-green) WITH texture.
    Right 45% = flat grey dialog (uniform = low variance).
    Size: 600 x 900 x 3.
    Boundary at x=495 (55% of 900).
    """
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
    """
    Synthetic FORMAT_B screenshot.
    Full-frame colorful aerial. No grey anywhere.
    Size: 600 x 900 x 3.
    """
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :, 0] = 60
    img[:, :, 1] = 120
    img[:, :, 2] = 180
    return img


@pytest.fixture(scope="module")
def format_a_with_bars() -> np.ndarray:
    """
    FORMAT_A with title bar and taskbar.

    Layout:
        [0:30]    = title bar (grey, 30px)
        [30:630]  = content (aerial + dialog, 600px)
        [630:660] = taskbar (grey, 30px)

    Total: 660 x 900 x 3.
    Content: 600 x 900 x 3.
    """
    img = np.zeros((660, 900, 3), dtype=np.uint8)
    img[:30, :, :] = 200
    img[30:630, :495, 0] = 60
    img[30:630, :495, 1] = 120
    img[30:630, :495, 2] = 180
    img[30:630, 495:, :] = 210
    img[630:, :, :] = 200
    return img


@pytest.fixture(scope="module")
def wide_image() -> np.ndarray:
    """
    Wide FORMAT_A: 1080 x 1920 x 3.
    Tests kernel scaling on large images.
    Boundary at x=1056 (55% of 1920).
    """
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[:, :1056, 0] = 60
    img[:, :1056, 1] = 120
    img[:, :1056, 2] = 180
    img[:, 1056:, :] = 210
    return img


@pytest.fixture(scope="module")
def tiny_image() -> np.ndarray:
    """Minimal valid image: 20 x 20 x 3."""
    return np.zeros((20, 20, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def result_a() -> DecompositionResult:
    """
    Pre-computed FORMAT_A decomposition result.
    Self-contained: does not depend on other fixtures.
    """
    d = ScreenshotDecomposer()
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :495, 0] = 60
    img[:, :495, 1] = 120
    img[:, :495, 2] = 180
    img[:, 495:, :] = 210
    return d.decompose(img)


@pytest.fixture(scope="module")
def result_b() -> DecompositionResult:
    """
    Pre-computed FORMAT_B decomposition result.
    Self-contained: does not depend on other fixtures.
    """
    d = ScreenshotDecomposer()
    img = np.zeros((600, 900, 3), dtype=np.uint8)
    img[:, :, 0] = 60
    img[:, :, 1] = 120
    img[:, :, 2] = 180
    return d.decompose(img)


@pytest.fixture
def real_screenshot() -> np.ndarray:
    """
    Real screenshot from data/fixtures/.
    Function-scoped: real I/O not cached.
    Skipped automatically if file not found.
    """
    path = "data/fixtures/sample_screenshot.png"
    img = cv2.imread(path)
    if img is None:
        pytest.skip(
            f"Real fixture not found: {path}"
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────
# _is_grey
# ─────────────────────────────────────────────────


class TestIsGrey:
    """
    Tests for _is_grey() boolean mask.

    Grey criteria:
        _GREY_LOW  = 160  (exclusive lower bound)
        _GREY_HIGH = 250  (exclusive upper bound)
        _GREY_DIFF = 20   (max R-G or G-B diff)
    """

    def test_pure_grey_all_true(self):
        """All-grey image → entire mask True."""
        img = np.full(
            (10, 10, 3), 200, dtype=np.uint8
        )
        assert _is_grey(img).all()

    def test_saturated_blue_false(self):
        """Pure blue → not grey."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 2] = 255
        assert not _is_grey(img).any()

    def test_too_dark_false(self):
        """Brightness 50 < _GREY_LOW → not grey."""
        img = np.full(
            (10, 10, 3), 50, dtype=np.uint8
        )
        assert not _is_grey(img).any()

    def test_too_bright_false(self):
        """Brightness 252 > _GREY_HIGH → not grey."""
        img = np.full(
            (10, 10, 3), 252, dtype=np.uint8
        )
        assert not _is_grey(img).any()

    def test_output_dtype_bool(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert _is_grey(img).dtype == bool

    def test_output_shape_h_w(self):
        """Shape must be (H, W) not (H, W, 3)."""
        img = np.zeros((15, 20, 3), dtype=np.uint8)
        assert _is_grey(img).shape == (15, 20)

    def test_no_uint8_overflow(self):
        """
        Critical overflow test.
        uint8: |R=10 - G=250| wraps → 16 (WRONG).
        int16: |10 - 250| = 240 (CORRECT → not grey).
        This test fails WITHOUT the int16 cast.
        """
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 250
        img[:, :, 2] = 200
        assert not _is_grey(img).any()

    def test_left_grey_right_colorful(self):
        """Mixed image: correct per-pixel split."""
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        img[:, :10, :] = 200
        img[:, 10:, 2] = 200
        mask = _is_grey(img)
        assert mask[:, :10].all()
        assert not mask[:, 10:].any()

    def test_boundary_inclusive_above_low(self):
        """Brightness 161 (just above 160) → grey."""
        img = np.full(
            (5, 5, 3), 161, dtype=np.uint8
        )
        assert _is_grey(img).all()

    def test_boundary_exclusive_at_low(self):
        """
        Brightness exactly 160 = _GREY_LOW.
        Condition: brightness > _GREY_LOW (strict).
        160 is NOT grey.
        """
        img = np.full(
            (5, 5, 3), 160, dtype=np.uint8
        )
        assert not _is_grey(img).any()


# ─────────────────────────────────────────────────
# _smooth1d
# ─────────────────────────────────────────────────


class TestSmooth1d:

    def test_constant_signal_unchanged(self):
        """Smoothing constant signal → same interior."""
        signal = np.ones(100, dtype=np.float32)
        result = _smooth1d(signal, 5)
        np.testing.assert_allclose(
            result[5:-5], signal[5:-5], atol=1e-5
        )

    def test_output_length_unchanged(self):
        signal = np.random.rand(200).astype(np.float32)
        assert len(_smooth1d(signal, 7)) == 200

    def test_spike_attenuated(self):
        """Single spike reduced in amplitude."""
        signal = np.zeros(100, dtype=np.float32)
        signal[50] = 100.0
        result = _smooth1d(signal, 11)
        assert 0.0 < result[50] < 100.0

    def test_spike_energy_spread(self):
        """Spike energy spreads to neighbours."""
        signal = np.zeros(100, dtype=np.float32)
        signal[50] = 100.0
        result = _smooth1d(signal, 11)
        assert result[46] > 0.0
        assert result[54] > 0.0

    def test_zero_signal_stays_zero(self):
        signal = np.zeros(50, dtype=np.float32)
        np.testing.assert_allclose(
            _smooth1d(signal, 5), 0.0
        )

    def test_output_is_float(self):
        signal = np.ones(50, dtype=np.float32)
        result = _smooth1d(signal, 5)
        assert result.dtype in (
            np.float32, np.float64
        )


# ─────────────────────────────────────────────────
# _kernel_size
# ─────────────────────────────────────────────────


class TestKernelSize:

    @pytest.mark.parametrize("width", [
        100, 200, 400, 800, 1000, 1920, 3840,
    ])
    def test_always_odd(self, width: int):
        """Kernel must always be odd."""
        k = _kernel_size(width)
        assert k % 2 == 1, (
            f"width={width} → even kernel={k}"
        )

    @pytest.mark.parametrize("width", [
        1, 5, 10, 50, 100,
    ])
    def test_minimum_five(self, width: int):
        assert _kernel_size(width) >= 5

    def test_monotone_wider_larger(self):
        assert (
            _kernel_size(400)
            < _kernel_size(800)
            < _kernel_size(1920)
        )


# ─────────────────────────────────────────────────
# _grey_profile
# ─────────────────────────────────────────────────


class TestGreyProfile:

    def test_output_shape_equals_width(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _grey_profile(img).shape == (200,)

    def test_output_dtype_float32(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _grey_profile(img).dtype == np.float32

    def test_values_bounded_zero_one(self):
        img = np.full(
            (50, 100, 3), 200, dtype=np.uint8
        )
        profile = _grey_profile(img)
        assert profile.min() >= 0.0
        assert profile.max() <= 1.0

    def test_all_grey_near_one(self):
        img = np.full(
            (100, 200, 3), 200, dtype=np.uint8
        )
        assert _grey_profile(img).mean() > 0.9

    def test_colorful_near_zero(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, :, 2] = 200
        assert _grey_profile(img).mean() < 0.1

    def test_right_higher_for_format_a(
        self, format_a: np.ndarray
    ):
        """FORMAT_A: right (dialog) greyer than left."""
        profile = _grey_profile(format_a)
        w = format_a.shape[1]
        assert (
            profile[w // 2:].mean()
            > profile[:w // 2].mean()
        )


# ─────────────────────────────────────────────────
# _edge_profile
# ─────────────────────────────────────────────────


class TestEdgeProfile:

    def test_output_shape_equals_width(
        self, format_a: np.ndarray
    ):
        assert (
            _edge_profile(format_a).shape
            == (format_a.shape[1],)
        )

    def test_output_dtype_float32(
        self, format_a: np.ndarray
    ):
        assert (
            _edge_profile(format_a).dtype
            == np.float32
        )

    def test_uniform_near_zero(self):
        img = np.full(
            (100, 200, 3), 128, dtype=np.uint8
        )
        assert _edge_profile(img).mean() < 5.0

    def test_non_negative(
        self, format_a: np.ndarray
    ):
        assert (_edge_profile(format_a) >= 0.0).all()

    def test_boundary_zone_strongest(
        self, format_a: np.ndarray
    ):
        """
        Boundary zone [35%, 70%] must have strongest
        edge. Outer regions [0-20%] and [80-100%]
        should have lower mean strength.
        """
        profile = _edge_profile(format_a)
        w = format_a.shape[1]
        zone = profile[
            int(w * 0.35): int(w * 0.70)
        ]
        outer = np.concatenate([
            profile[: int(w * 0.20)],
            profile[int(w * 0.80):],
        ])
        assert zone.max() > outer.mean()


# ─────────────────────────────────────────────────
# _variance_profile
# ─────────────────────────────────────────────────


class TestVarianceProfile:

    def test_output_shape_equals_width(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _variance_profile(img).shape == (200,)

    def test_output_dtype_float32(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        assert (
            _variance_profile(img).dtype == np.float32
        )

    def test_uniform_near_zero(self):
        img = np.full(
            (100, 200, 3), 128, dtype=np.uint8
        )
        assert _variance_profile(img).mean() < 5.0

    def test_random_high_variance(self):
        rng = np.random.default_rng(42)
        img = rng.integers(
            0, 255, (100, 200, 3), dtype=np.uint8
        )
        assert (
            _variance_profile(img)[10:-10].mean()
            > 10.0
        )

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        img = rng.integers(
            0, 255, (50, 100, 3), dtype=np.uint8
        )
        assert (_variance_profile(img) >= 0.0).all()

    def test_too_narrow_returns_zeros(self):
        """
        Width=4 <= 2*_VARIANCE_BLOCK=6.
        No valid windows → zeros, no crash.
        """
        img = np.zeros((100, 4, 3), dtype=np.uint8)
        result = _variance_profile(img)
        assert result.shape == (4,)
        assert (result == 0.0).all()

    def test_aerial_higher_than_dialog(
        self, format_a: np.ndarray
    ):
        """Colorful aerial > grey dialog variance."""
        profile = _variance_profile(format_a)
        assert profile[:400].mean() > profile[550:].mean()

    def test_tiny_image_no_crash(
        self, tiny_image: np.ndarray
    ):
        result = _variance_profile(tiny_image)
        assert result.shape == (tiny_image.shape[1],)


# ─────────────────────────────────────────────────
# _validate_image
# ─────────────────────────────────────────────────


class TestValidateImage:

    def test_valid_no_raise(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        _validate_image(img, "test")

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            _validate_image(None, "fn")

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError):
            _validate_image([[1, 2, 3]], "fn")

    def test_2d_raises_value_error(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(
            ValueError, match="shape"
        ):
            _validate_image(img, "fn")

    def test_4channel_raises_value_error(self):
        img = np.zeros((100, 200, 4), dtype=np.uint8)
        with pytest.raises(
            ValueError, match="shape"
        ):
            _validate_image(img, "fn")

    def test_1channel_raises_value_error(self):
        img = np.zeros((100, 200, 1), dtype=np.uint8)
        with pytest.raises(
            ValueError, match="shape"
        ):
            _validate_image(img, "fn")

    def test_empty_raises_value_error(self):
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(
            ValueError, match="empty"
        ):
            _validate_image(img, "fn")

    def test_caller_name_in_error(self):
        with pytest.raises(TypeError) as exc:
            _validate_image(None, "my_function")
        assert "my_function" in str(exc.value)

    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, np.int32,
    ])
    def test_non_uint8_passes(self, dtype):
        """Dtype enforcement is caller's responsibility."""
        img = np.zeros((10, 10, 3), dtype=dtype)
        _validate_image(img, "test")


# ─────────────────────────────────────────────────
# detect_format
# ─────────────────────────────────────────────────


class TestDetectFormat:

    def test_format_a_detected(
        self, decomposer, format_a
    ):
        fmt, _ = decomposer.detect_format(format_a)
        assert fmt == 'FORMAT_A'

    def test_format_b_detected(
        self, decomposer, format_b
    ):
        fmt, _ = decomposer.detect_format(format_b)
        assert fmt == 'FORMAT_B'

    def test_returns_valid_string(
        self, decomposer, format_a
    ):
        fmt, _ = decomposer.detect_format(format_a)
        assert fmt in ('FORMAT_A', 'FORMAT_B')

    def test_confidence_in_range(
        self, decomposer, format_a
    ):
        _, conf = decomposer.detect_format(format_a)
        assert 0.0 <= conf <= 1.0

    def test_confidence_capped_at_one(
        self, decomposer
    ):
        img = np.full(
            (100, 200, 3), 200, dtype=np.uint8
        )
        _, conf = decomposer.detect_format(img)
        assert conf <= 1.0

    def test_format_a_high_confidence(
        self, decomposer, format_a
    ):
        _, conf = decomposer.detect_format(format_a)
        assert conf > 0.5

    def test_wide_image_format_a(
        self, decomposer, wide_image
    ):
        fmt, conf = decomposer.detect_format(wide_image)
        assert fmt == 'FORMAT_A'
        assert conf > 0.5

    def test_none_raises_type_error(self, decomposer):
        with pytest.raises(TypeError):
            decomposer.detect_format(None)

    def test_2d_raises_value_error(self, decomposer):
        img = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            decomposer.detect_format(img)


# ─────────────────────────────────────────────────
# find_dialog_boundary
# ─────────────────────────────────────────────────


class TestFindDialogBoundary:

    def test_returns_two_tuple(
        self, decomposer, format_a
    ):
        result = decomposer.find_dialog_boundary(
            format_a
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_boundary_x_is_int(
        self, decomposer, format_a
    ):
        bx, _ = decomposer.find_dialog_boundary(
            format_a
        )
        assert isinstance(bx, int)

    def test_confidence_is_float(
        self, decomposer, format_a
    ):
        _, conf = decomposer.find_dialog_boundary(
            format_a
        )
        assert isinstance(conf, float)

    def test_boundary_in_valid_range(
        self, decomposer, format_a
    ):
        w = format_a.shape[1]
        bx, _ = decomposer.find_dialog_boundary(
            format_a
        )
        assert w * 0.35 <= bx <= w * 0.70

    def test_confidence_minimum_0_3(
        self, decomposer, format_a
    ):
        _, conf = decomposer.find_dialog_boundary(
            format_a
        )
        assert conf >= 0.3

    def test_confidence_maximum_1_0(
        self, decomposer, format_a
    ):
        _, conf = decomposer.find_dialog_boundary(
            format_a
        )
        assert conf <= 1.0

    def test_dialog_minimum_80px(
        self, decomposer, format_a
    ):
        w = format_a.shape[1]
        bx, _ = decomposer.find_dialog_boundary(
            format_a
        )
        assert (w - bx) >= 80

    def test_on_bar_cropped_content(
        self, decomposer, format_a_with_bars
    ):
        """Works correctly on bar-cropped content."""
        content = format_a_with_bars[30:630, :, :]
        w = content.shape[1]
        bx, conf = decomposer.find_dialog_boundary(
            content
        )
        assert w * 0.35 <= bx <= w * 0.70
        assert conf >= 0.3

    def test_none_raises_type_error(self, decomposer):
        with pytest.raises(TypeError):
            decomposer.find_dialog_boundary(None)

    def test_2d_raises_value_error(self, decomposer):
        img = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            decomposer.find_dialog_boundary(img)


# ─────────────────────────────────────────────────
# detect_bars
# ─────────────────────────────────────────────────


class TestDetectBars:

    def test_returns_two_tuple(
        self, decomposer, format_b
    ):
        result = decomposer.detect_bars(format_b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_bars_in_colorful_image(
        self, decomposer, format_b
    ):
        """
        format_b = fully colorful, no grey strips.
        format_a not used: its grey dialog could
        trigger false bar detection.
        """
        title, taskbar = decomposer.detect_bars(
            format_b
        )
        assert title is None
        assert taskbar is None

    def test_title_bar_detected(
        self, decomposer, format_a_with_bars
    ):
        title, _ = decomposer.detect_bars(
            format_a_with_bars
        )
        assert title is not None
        assert isinstance(title, BarInfo)

    def test_taskbar_detected(
        self, decomposer, format_a_with_bars
    ):
        _, taskbar = decomposer.detect_bars(
            format_a_with_bars
        )
        assert taskbar is not None
        assert isinstance(taskbar, BarInfo)

    def test_title_bar_y_zero(
        self, decomposer, format_a_with_bars
    ):
        title, _ = decomposer.detect_bars(
            format_a_with_bars
        )
        assert title.y == 0

    def test_title_bar_height_positive(
        self, decomposer, format_a_with_bars
    ):
        title, _ = decomposer.detect_bars(
            format_a_with_bars
        )
        assert title.height > 0

    def test_taskbar_y_near_bottom(
        self, decomposer, format_a_with_bars
    ):
        h = format_a_with_bars.shape[0]
        _, taskbar = decomposer.detect_bars(
            format_a_with_bars
        )
        assert taskbar.y > h * 0.8

    def test_taskbar_height_positive(
        self, decomposer, format_a_with_bars
    ):
        _, taskbar = decomposer.detect_bars(
            format_a_with_bars
        )
        assert taskbar.height > 0

    def test_none_raises_type_error(self, decomposer):
        with pytest.raises(TypeError):
            decomposer.detect_bars(None)


# ─────────────────────────────────────────────────
# decompose (End-to-End)
# ─────────────────────────────────────────────────


class TestDecompose:

    def test_returns_decomposition_result(
        self, result_a
    ):
        assert isinstance(
            result_a, DecompositionResult
        )

    def test_format_a_correct(self, result_a):
        assert result_a.format == 'FORMAT_A'

    def test_format_b_correct(self, result_b):
        assert result_b.format == 'FORMAT_B'

    def test_format_a_has_dialog(self, result_a):
        assert result_a.has_dialog()
        assert result_a.dialog is not None

    def test_format_b_no_dialog(self, result_b):
        assert not result_b.has_dialog()
        assert result_b.dialog is None

    def test_aerial_not_none(self, result_a):
        assert result_a.aerial is not None

    def test_aerial_is_ndarray(self, result_a):
        assert isinstance(
            result_a.aerial, np.ndarray
        )

    def test_aerial_3_channel(self, result_a):
        assert result_a.aerial.ndim == 3
        assert result_a.aerial.shape[2] == 3

    def test_dialog_3_channel(self, result_a):
        assert result_a.dialog.ndim == 3
        assert result_a.dialog.shape[2] == 3

    def test_aerial_narrower_than_full_width(
        self, result_a
    ):
        """FORMAT_A aerial < full 900px width."""
        assert result_a.aerial_width() < 900

    def test_aerial_dialog_same_height(
        self, result_a
    ):
        """No pixels lost in height dimension."""
        assert (
            result_a.aerial.shape[0]
            == result_a.dialog.shape[0]
        )

    def test_aerial_dialog_widths_sum_to_900(
        self, result_a
    ):
        """
        aerial + dialog = 900px.
        No pixels lost or duplicated at boundary.
        format_a has no bars so content = 900px wide.
        """
        total = (
            result_a.aerial_width()
            + result_a.dialog.shape[1]
        )
        assert total == 900

    def test_format_confidence_in_range(
        self, result_a
    ):
        assert (
            0.0 <= result_a.format_confidence <= 1.0
        )

    def test_boundary_confidence_minimum_0_3(
        self, result_a
    ):
        assert result_a.boundary_confidence >= 0.3

    def test_format_b_boundary_x_none(
        self, result_b
    ):
        assert result_b.boundary_x is None

    def test_format_b_boundary_confidence_zero(
        self, result_b
    ):
        assert result_b.boundary_confidence == 0.0

    def test_format_a_boundary_x_int(
        self, result_a
    ):
        assert isinstance(result_a.boundary_x, int)

    def test_aerial_bbox_4_elements(self, result_a):
        assert len(result_a.aerial_bbox) == 4

    def test_aerial_bbox_non_negative(
        self, result_a
    ):
        assert all(
            v >= 0 for v in result_a.aerial_bbox
        )

    def test_bars_excluded_from_height(
        self, decomposer, format_a_with_bars
    ):
        result = decomposer.decompose(
            format_a_with_bars
        )
        assert result.aerial_height() < 660

    def test_bars_correct_content_height(
        self, decomposer, format_a_with_bars
):
        """
        Content height = total - detected_title - detected_taskbar.
        Uses actual detected heights, not hardcoded 30px.
        """
        result = decomposer.decompose(
            format_a_with_bars
        )
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

    def test_stateless_same_result_twice(
        self, decomposer, format_a
    ):
        """Stateless: identical output on repeat."""
        r1 = decomposer.decompose(format_a)
        r2 = decomposer.decompose(format_a)
        assert r1.format == r2.format
        assert r1.boundary_x == r2.boundary_x
        assert (
            r1.format_confidence
            == r2.format_confidence
        )

    def test_wide_image(
        self, decomposer, wide_image
    ):
        result = decomposer.decompose(wide_image)
        assert result.format == 'FORMAT_A'
        assert result.aerial.size > 0

    def test_repr_concise(self, result_a):
        """__repr__ must not print array contents."""
        r = repr(result_a)
        assert "DecompositionResult" in r
        assert "FORMAT_A" in r
        assert len(r) < 300

    def test_none_raises_type_error(
        self, decomposer
    ):
        with pytest.raises(TypeError):
            decomposer.decompose(None)

    def test_2d_raises_value_error(self, decomposer):
        img = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            decomposer.decompose(img)

    def test_empty_raises_value_error(
        self, decomposer
    ):
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            decomposer.decompose(img)


# ─────────────────────────────────────────────────
# DecompositionResult Helpers
# ─────────────────────────────────────────────────


class TestDecompositionResultHelpers:

    def test_has_dialog_true(self, result_a):
        assert result_a.has_dialog() is True

    def test_has_dialog_false(self, result_b):
        assert result_b.has_dialog() is False

    def test_aerial_width_matches_array(
        self, result_a
    ):
        assert (
            result_a.aerial_width()
            == result_a.aerial.shape[1]
        )

    def test_aerial_height_matches_array(
        self, result_a
    ):
        assert (
            result_a.aerial_height()
            == result_a.aerial.shape[0]
        )

    def test_aerial_fraction_between_0_1(
        self, result_a
    ):
        frac = result_a.aerial_fraction(900)
        assert 0.0 < frac < 1.0

    def test_aerial_fraction_format_b_near_one(
        self, result_b
    ):
        """
        FORMAT_B: aerial = full content.
        No bars in fixture → fraction > 0.9.
        """
        frac = result_b.aerial_fraction(900)
        assert frac > 0.9

    def test_aerial_fraction_zero_raises(
        self, result_a
    ):
        with pytest.raises(ValueError):
            result_a.aerial_fraction(0)

    def test_aerial_fraction_negative_raises(
        self, result_a
    ):
        with pytest.raises(ValueError):
            result_a.aerial_fraction(-1)


# ─────────────────────────────────────────────────
# BarInfo
# ─────────────────────────────────────────────────


class TestBarInfo:

    def test_fields_readable(self):
        bar = BarInfo(y=0, height=30)
        assert bar.y == 0
        assert bar.height == 30

    def test_y_immutable(self):
        bar = BarInfo(y=0, height=30)
        with pytest.raises(
            dataclasses.FrozenInstanceError
        ):
            bar.y = 10  # type: ignore

    def test_height_immutable(self):
        bar = BarInfo(y=0, height=30)
        with pytest.raises(
            dataclasses.FrozenInstanceError
        ):
            bar.height = 99  # type: ignore

    def test_equality_same_values(self):
        assert (
            BarInfo(y=0, height=30)
            == BarInfo(y=0, height=30)
        )

    def test_inequality_different_y(self):
        assert (
            BarInfo(y=5, height=30)
            != BarInfo(y=0, height=30)
        )

    def test_inequality_different_height(self):
        assert (
            BarInfo(y=0, height=25)
            != BarInfo(y=0, height=30)
        )


# ─────────────────────────────────────────────────
# REAL IMAGE
# ─────────────────────────────────────────────────


class TestRealImage:
    """
    Integration tests on real dataset screenshots.
    Skipped automatically if fixture not found.
    """

    def test_no_crash(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert result is not None

    def test_aerial_non_empty(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert result.aerial.size > 0

    def test_valid_format_string(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert result.format in (
            'FORMAT_A', 'FORMAT_B'
        )

    def test_confidence_valid(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert (
            0.0 <= result.format_confidence <= 1.0
        )

    def test_aerial_3_channel(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert result.aerial.shape[2] == 3

    def test_repr_concise(
        self, decomposer, real_screenshot
    ):
        result = decomposer.decompose(real_screenshot)
        assert len(repr(result)) < 300

    def test_stateless_on_real(
        self, decomposer, real_screenshot
    ):
        """Identical result on repeat call."""
        r1 = decomposer.decompose(real_screenshot)
        r2 = decomposer.decompose(real_screenshot)
        assert r1.format == r2.format
        assert r1.boundary_x == r2.boundary_x