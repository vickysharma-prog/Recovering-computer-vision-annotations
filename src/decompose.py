"""
Screenshot decomposition for bird annotation recovery.

Splits aerial survey screenshots into two regions:
    - Aerial photograph (contains annotation dots)
    - Dialog box (contains species legend), if present

Pipeline position: Stage 1 of 7
Input: Raw screenshot (RGB numpy array)
Output: DecompositionResult dataclass

Boundary detection uses 3-method consensus:
    1. Grey profile — first grey column
    2. Sobel edges — strongest vertical edge
    3. Variance drop — first low-variance column
Final boundary = median of 3 candidates.
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
    """Load decomposition config from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)["decompose"]


_CONFIG = _load_config()


# ─────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class BarInfo:
    """
    Detected UI bar metadata.

    Attributes:
        y: Top y coordinate in original image
        height: Bar height in pixels
    """
    y: int
    height: int


@dataclass
class DecompositionResult:
    """
    Result of screenshot decomposition.

    Attributes:
        has_dialog: True if dialog box present
        detection_confidence: Confidence in dialog detection [0,1]
        aerial: Aerial region (H,W,3) RGB
        dialog: Dialog region (H,W,3) or None
        boundary_x: Dialog start x-pixel or None
        boundary_confidence: Boundary location confidence [0,1]
        aerial_bbox: (x,y,w,h) in original image coords
        title_bar: Title bar info or None
        taskbar: Taskbar info or None
    """
    has_dialog: bool
    detection_confidence: float
    aerial: np.ndarray
    dialog: Optional[np.ndarray]
    boundary_x: Optional[int]
    boundary_confidence: float
    aerial_bbox: tuple[int, int, int, int]
    title_bar: Optional[BarInfo]
    taskbar: Optional[BarInfo]

    def __repr__(self) -> str:
        dialog_shape = (
            self.dialog.shape
            if self.dialog is not None
            else None
        )
        return (
            f"DecompositionResult("
            f"has_dialog={self.has_dialog}, "
            f"confidence={self.detection_confidence:.3f}, "
            f"aerial={self.aerial.shape}, "
            f"dialog={dialog_shape}, "
            f"boundary_x={self.boundary_x})"
        )

    def aerial_width(self) -> int:
        """Width of aerial region in pixels."""
        return int(self.aerial.shape[1])

    def aerial_height(self) -> int:
        """Height of aerial region in pixels."""
        return int(self.aerial.shape[0])


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────

def _is_grey(img_rgb: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True where pixels are grey.

    Grey defined as R ≈ G ≈ B within threshold,
    brightness in configured range.

    Args:
        img_rgb: RGB image (H, W, 3) uint8

    Returns:
        Boolean mask (H, W)
    """
    r = img_rgb[:, :, 0].astype(np.int16)
    g = img_rgb[:, :, 1].astype(np.int16)
    b = img_rgb[:, :, 2].astype(np.int16)
    brightness = img_rgb[:, :, 0]

    return (
        (np.abs(r - g) < _CONFIG["grey_diff"]) &
        (np.abs(g - b) < _CONFIG["grey_diff"]) &
        (brightness > _CONFIG["grey_low"]) &
        (brightness < _CONFIG["grey_high"])
    )


def _smooth1d(signal: np.ndarray, ks: int) -> np.ndarray:
    """1D moving average smoothing."""
    return np.convolve(
        signal, np.ones(ks) / ks, mode='same'
    )


def _kernel_size(image_width: int) -> int:
    """Compute odd smoothing kernel size from image width."""
    k = max(image_width // 50, 5)
    return k if k % 2 == 1 else k + 1


def _grey_profile(img_rgb: np.ndarray) -> np.ndarray:
    """Per-column grey pixel fraction."""
    return _is_grey(img_rgb).mean(axis=0).astype(np.float32)


def _edge_profile(img_rgb: np.ndarray) -> np.ndarray:
    """Per-column Sobel edge strength."""
    h = img_rgb.shape[0]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobel = np.abs(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    )
    y1 = int(h * _CONFIG["edge_margin_top"])
    y2 = int(h * _CONFIG["edge_margin_bot"])
    return sobel[y1:y2, :].mean(axis=0).astype(np.float32)


def _variance_profile(img_rgb: np.ndarray) -> np.ndarray:
    """
    Per-column color variance over middle region.

    Vectorized via sliding_window_view (NumPy >= 1.20).
    Falls back to loop on older NumPy.
    """
    h, w = img_rgb.shape[:2]
    y1 = int(h * _CONFIG["variance_strip_top"])
    y2 = int(h * _CONFIG["variance_strip_bot"])
    strip = img_rgb[y1:y2, :, :].astype(np.float32)

    b = _CONFIG["variance_block"]
    var = np.zeros(w, dtype=np.float32)

    if w <= 2 * b:
        return var

    try:
        per_ch = np.zeros((3, w), dtype=np.float32)
        for c in range(3):
            wins = np.lib.stride_tricks.sliding_window_view(
                strip[:, :, c],
                window_shape=2 * b,
                axis=1,
            )
            per_ch[c, b: b + wins.shape[1]] = (
                wins.std(axis=(0, 2))
            )
        var = per_ch.mean(axis=0)

    except AttributeError:
        logger.debug(
            "sliding_window_view unavailable, using fallback"
        )
        for x in range(b, w - b):
            block = strip[:, x - b: x + b + 1, :]
            var[x] = float(block.std())

    return var


def _validate_image(
    img_rgb: object, caller: str
) -> None:
    """Validate (H, W, 3) RGB numpy array."""
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError(
            f"{caller}(): expected np.ndarray, "
            f"got {type(img_rgb).__name__}"
        )
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(
            f"{caller}(): expected (H, W, 3), "
            f"got shape {img_rgb.shape}"
        )
    if img_rgb.size == 0:
        raise ValueError(
            f"{caller}(): received empty array"
        )


# ─────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────

class ScreenshotDecomposer:
    """
    Splits aerial survey screenshots into aerial + dialog.

    The annotation tool baked colored dots into screenshots
    alongside a species count dialog. This class separates
    the two regions for independent processing.

    Thread safety: Stateless, safe to reuse.

    Example:
        >>> decomposer = ScreenshotDecomposer()
        >>> result = decomposer.decompose(img_rgb)
        >>> aerial = result.aerial
    """

    def decompose(
        self,
        img_rgb: np.ndarray,
        expect_dialog: bool | None = None,
    ) -> DecompositionResult:
        """
        Decompose screenshot into aerial + dialog regions.

        Args:
            img_rgb: Full screenshot (H,W,3) uint8 RGB
            expect_dialog: If True/False, skip auto-detection
                            and use provided value directly.
                            If None (default), auto-detect from image.

        Returns:
            DecompositionResult

        Raises:
            TypeError: img_rgb not np.ndarray
            ValueError: Wrong shape or empty
        """
        _validate_image(img_rgb, "decompose")

        h, w = img_rgb.shape[:2]
        title_bar, taskbar = self.detect_bars(img_rgb)

        y_start = title_bar.height if title_bar else 0
        y_end = taskbar.y if taskbar else h
        content = img_rgb[y_start:y_end, :, :]

        if expect_dialog is not None:
            has_dialog = expect_dialog
            conf = 1.0
        else:
            has_dialog, conf = self._detect_dialog_presence(
                img_rgb
            )

        logger.info(
            "decompose | %dx%d | has_dialog=%s conf=%.2f",
            w, h, has_dialog, conf,
        )

        if has_dialog:
            return self._split_with_dialog(
                content, conf,
                title_bar, taskbar,
                y_start, y_end, w,
            )
        return self._build_no_dialog(
            content, conf,
            title_bar, taskbar,
            y_start, y_end, w,
        )

    def _detect_dialog_presence(
        self, img_rgb: np.ndarray
    ) -> tuple[bool, float]:
        """Auto-detect if dialog box is present."""
        w = img_rgb.shape[1]
        right = img_rgb[
            :, int(w * _CONFIG["right_strip_start"]):, :
        ]
        grey_pct = float(_is_grey(right).mean() * 100.0)

        if grey_pct > _CONFIG["grey_pct_threshold"]:
            conf = min(grey_pct / 50.0, 1.0)
            logger.debug("Dialog detected: grey=%.1f%%", grey_pct)
            return True, conf

        conf = min((50.0 - grey_pct) / 50.0, 1.0)
        logger.debug("No dialog: grey=%.1f%%", grey_pct)
        return False, conf

    def find_dialog_boundary(
        self, img_rgb: np.ndarray
    ) -> tuple[int, float]:
        """
        Find x where aerial ends, dialog begins.

        Uses 3-method consensus (grey + edge + variance).
        Boundary constrained to [35%, 70%] of width.

        Returns:
            (boundary_x, confidence)
        """
        _validate_image(img_rgb, "find_dialog_boundary")

        w = img_rgb.shape[1]
        ks = _kernel_size(w)

        m1 = self._boundary_grey(img_rgb, w, ks)
        m2 = self._boundary_edge(img_rgb, w)
        m3 = self._boundary_variance(img_rgb, w, ks)

        candidates = [m1, m2, m3]
        boundary_x = int(np.median(candidates))

        logger.debug(
            "boundary candidates: grey=%d edge=%d var=%d → %d",
            m1, m2, m3, boundary_x,
        )

        return self._validate_boundary(
            boundary_x, candidates, w
        )

    def detect_bars(
        self, img_rgb: np.ndarray
    ) -> tuple[Optional[BarInfo], Optional[BarInfo]]:
        """
        Detect title bar and taskbar.

        Returns:
            (title_bar, taskbar) BarInfo or None each
        """
        _validate_image(img_rgb, "detect_bars")
        h = img_rgb.shape[0]
        return (
            self._find_top_bar(img_rgb),
            self._find_bottom_bar(img_rgb, h),
        )

    # ─────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────

    def _boundary_grey(
        self, img_rgb: np.ndarray, w: int, ks: int
    ) -> int:
        """Method 1: First grey column above threshold."""
        profile = _smooth1d(_grey_profile(img_rgb), ks)
        lo = int(w * _CONFIG["boundary_min"])
        hi = int(w * _CONFIG["boundary_max"])
        for x in range(lo, hi):
            window = profile[x: min(x + 20, hi)]
            if (
                len(window) >= _CONFIG["min_window_size"]
                and window.mean() > _CONFIG["grey_profile_threshold"]
            ):
                return x
        return w // 2

    def _boundary_edge(
        self, img_rgb: np.ndarray, w: int
    ) -> int:
        """Method 2: Strongest vertical edge."""
        profile = _edge_profile(img_rgb)
        lo = int(w * _CONFIG["boundary_min"])
        hi = int(w * _CONFIG["boundary_max"])
        segment = profile[lo:hi]
        if segment.size == 0:
            return w // 2
        return lo + int(np.argmax(segment))

    def _boundary_variance(
        self, img_rgb: np.ndarray, w: int, ks: int
    ) -> int:
        """Method 3: First low-variance column."""
        profile = _smooth1d(_variance_profile(img_rgb), ks)
        if profile.max() == 0.0:
            return w // 2

        lo = int(w * _CONFIG["boundary_min"])
        hi = int(w * _CONFIG["boundary_max"])
        aerial_mean = profile[
            int(w * _CONFIG["aerial_sample_start"]):
            int(w * _CONFIG["aerial_sample_end"])
        ].mean()
        threshold = aerial_mean * _CONFIG["variance_drop_ratio"]

        for x in range(lo, hi):
            window = profile[x: min(x + 15, hi)]
            if (
                len(window) >= _CONFIG["min_window_size"]
                and window.mean() < threshold
            ):
                return x
        return w // 2

    @staticmethod
    def _validate_boundary(
        boundary_x: int,
        candidates: list[int],
        w: int,
    ) -> tuple[int, float]:
        """Validate boundary and compute confidence."""
        out_of_range = not (
            w * _CONFIG["boundary_min"]
            <= boundary_x
            <= w * _CONFIG["boundary_max"]
        )
        too_narrow = (
            (w - boundary_x) < _CONFIG["min_dialog_width_px"]
        )

        if out_of_range or too_narrow:
            logger.warning(
                "Invalid boundary %d, using center fallback",
                boundary_x,
            )
            return w // 2, _CONFIG["low_confidence"]

        spread = float(max(candidates) - min(candidates))
        confidence = max(
            _CONFIG["low_confidence"],
            1.0 - spread / (w * 0.15),
        )
        return boundary_x, round(confidence, 3)

    @staticmethod
    def _find_top_bar(
        img_rgb: np.ndarray,
    ) -> Optional[BarInfo]:
        """Detect title bar at image top."""
        for bar_h in _CONFIG["title_bar_heights"]:
            strip = img_rgb[:bar_h, :, :]
            grey_pct = float(_is_grey(strip).mean() * 100)
            if grey_pct > _CONFIG["title_grey_threshold"]:
                logger.debug("Title bar: h=%d", bar_h)
                return BarInfo(y=0, height=bar_h)
        return None

    @staticmethod
    def _find_bottom_bar(
        img_rgb: np.ndarray, h: int
    ) -> Optional[BarInfo]:
        """Detect taskbar at image bottom."""
        for bar_h in _CONFIG["taskbar_heights"]:
            strip = img_rgb[h - bar_h:, :, :]
            grey_pct = float(_is_grey(strip).mean() * 100)
            if grey_pct > _CONFIG["taskbar_grey_threshold"]:
                logger.debug("Taskbar: h=%d", bar_h)
                return BarInfo(y=h - bar_h, height=bar_h)
        return None

    def _split_with_dialog(
        self,
        content: np.ndarray,
        conf: float,
        title_bar: Optional[BarInfo],
        taskbar: Optional[BarInfo],
        y_start: int,
        y_end: int,
        orig_w: int,
    ) -> DecompositionResult:
        """Build result when dialog is present."""
        bx, bx_conf = self.find_dialog_boundary(content)
        aerial = content[:, :bx, :]
        dialog = content[:, bx:, :]

        logger.info(
            "With dialog: aerial=%dx%d dialog=%dx%d",
            aerial.shape[1], aerial.shape[0],
            dialog.shape[1], dialog.shape[0],
        )

        return DecompositionResult(
            has_dialog=True,
            detection_confidence=round(conf, 3),
            aerial=aerial,
            dialog=dialog,
            boundary_x=bx,
            boundary_confidence=bx_conf,
            aerial_bbox=(0, y_start, bx, y_end - y_start),
            title_bar=title_bar,
            taskbar=taskbar,
        )

    @staticmethod
    def _build_no_dialog(
        content: np.ndarray,
        conf: float,
        title_bar: Optional[BarInfo],
        taskbar: Optional[BarInfo],
        y_start: int,
        y_end: int,
        orig_w: int,
    ) -> DecompositionResult:
        """Build result when no dialog present."""
        logger.info(
            "No dialog: aerial=%dx%d",
            content.shape[1], content.shape[0],
        )

        return DecompositionResult(
            has_dialog=False,
            detection_confidence=round(conf, 3),
            aerial=content,
            dialog=None,
            boundary_x=None,
            boundary_confidence=0.0,
            aerial_bbox=(
                0, y_start, orig_w, y_end - y_start
            ),
            title_bar=title_bar,
            taskbar=taskbar,
        )