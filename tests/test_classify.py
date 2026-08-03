from __future__ import annotations

import glob

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pytest

from src.legend import LegendEntry, canonical_template, parse_screenshot
from src.classify import (
    AerialDot,
    detect_dots,
    assign_classes,
    class_counts,
    select_by_count,
    _template_similarity,
    _split_cluster,
    _dot_centers,
)


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _circle_template() -> np.ndarray:
    m = np.zeros((24, 24), np.uint8)
    cv2.circle(m, (12, 12), 9, 1, -1)
    return canonical_template(m)


def _square_template() -> np.ndarray:
    return canonical_template(np.ones((20, 20), np.uint8))


def _triangle_template() -> np.ndarray:
    m = np.zeros((24, 24), np.uint8)
    pts = np.array([[12, 3], [3, 20], [21, 20]], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return canonical_template(m)


def _aerial_with_dots(centers, color=(255, 30, 30)) -> np.ndarray:
    """Low-saturation grey aerial with bright dots (realistic background)."""
    img = np.full((100, 200, 3), 120, np.uint8)
    for (cx, cy) in centers:
        cv2.circle(img, (cx, cy), 5, color, -1)
    return img


# ─────────────────────────────────────────────────
# TEMPLATE SIMILARITY
# ─────────────────────────────────────────────────

class TestTemplateSimilarity:

    def test_identical_is_one(self):
        a = np.ones((24, 24), np.float32)
        assert _template_similarity(a, a) == pytest.approx(1.0)

    def test_zero_norm_is_zero(self):
        a = np.ones((24, 24), np.float32)
        z = np.zeros((24, 24), np.float32)
        assert _template_similarity(a, z) == 0.0

    def test_circle_more_similar_to_circle_than_square(self):
        c, s = _circle_template(), _square_template()
        assert _template_similarity(c, c) > _template_similarity(c, s)


# ─────────────────────────────────────────────────
# CLUSTER SPLITTING
# ─────────────────────────────────────────────────

class TestSplitCluster:

    def test_two_merged_disks_split(self):
        m = np.zeros((20, 40), np.uint8)
        cv2.circle(m, (12, 10), 7, 1, -1)
        cv2.circle(m, (26, 10), 7, 1, -1)
        assert len(_split_cluster(m, 2)) == 2

    def test_single_disk_one_center(self):
        m = np.zeros((20, 20), np.uint8)
        cv2.circle(m, (10, 10), 6, 1, -1)
        assert len(_split_cluster(m, 1)) == 1


# ─────────────────────────────────────────────────
# DOT DETECTION
# ─────────────────────────────────────────────────

class TestDetectDots:

    def test_counts_isolated_dots(self):
        centers, _ = _dot_centers(_aerial_with_dots([(30, 50), (90, 50), (150, 50)]))
        assert len(centers) == 3

    def test_no_dots_empty(self):
        img = np.full((100, 200, 3), 120, np.uint8)
        assert detect_dots(img) == []

    def test_detect_dots_reads_color(self):
        dots = detect_dots(_aerial_with_dots([(30, 50), (90, 50)]))
        assert len(dots) == 2
        assert all(d.color == "red" for d in dots)

    def test_exclude_box_drops_dots(self):
        img = _aerial_with_dots([(30, 50), (90, 50), (150, 50)])
        dots = detect_dots(img, exclude=(0, 0, 50, 100))
        assert len(dots) == 2  # left dot excluded

    def test_dots_have_quality(self):
        dots = detect_dots(_aerial_with_dots([(90, 50)]))
        assert dots[0].quality > 0.0


# ─────────────────────────────────────────────────
# CLASS ASSIGNMENT
# ─────────────────────────────────────────────────

# Colour vectors (Lab-ish) far enough apart that RED and BLUE fall outside the
# palette-anchoring reject radius. Assignment now keys on `color_vec`, not the
# `color` name string.
_RED_VEC = np.array([130.0, 185.0, 170.0], np.float32)
_BLUE_VEC = np.array([90.0, 130.0, 60.0], np.float32)


def _entry(color, shape, template, name, vec=_RED_VEC) -> LegendEntry:
    e = LegendEntry(
        row=0, cy=0.0, cx=0.0, shape=shape, color=color, hue=0.0,
        marker=np.zeros((2, 2, 3), np.uint8), template=template,
        class_name=name, species=name.split()[0], category=shape,
    )
    e._color_vec = vec        # pre-seed palette cache (marker is a stub here)
    e._shape_tmpl = template  # pre-seed NCC shape template (marker is a stub)
    return e


class TestAssignClasses:

    def test_single_color_candidate(self):
        e = _entry("red", "circle", _circle_template(), "BRPE WBN")
        d = AerialDot(cx=0, cy=0, color="red", shape="circle",
                      area=10, template=_circle_template(), color_vec=_RED_VEC)
        assign_classes([d], [e])
        assert d.class_name == "BRPE WBN"
        assert d.match_score == 1.0

    def test_shape_breaks_same_color_tie(self):
        # Same colour, different shape -> template (NCC) breaks the tie.
        # Non-uniform shapes (circle vs triangle) so mean-subtracted NCC is
        # well-defined (a solid square canonical is uniform -> degenerate).
        e1 = _entry("red", "circle", _circle_template(), "RED CIRCLE")
        e2 = _entry("red", "triangle", _triangle_template(), "RED TRIANGLE")
        d = AerialDot(cx=0, cy=0, color="red", shape="triangle",
                      area=10, template=_triangle_template(), color_vec=_RED_VEC)
        assign_classes([d], [e1, e2])
        assert d.class_name == "RED TRIANGLE"

    def test_color_absent_from_legend_unassigned(self):
        # Dot colour is far from the only legend colour -> off-palette reject.
        e = _entry("red", "circle", _circle_template(), "BRPE WBN")
        d = AerialDot(cx=0, cy=0, color="blue", shape="circle",
                      area=10, template=_circle_template(), color_vec=_BLUE_VEC)
        assign_classes([d], [e])
        assert d.class_name is None


# ─────────────────────────────────────────────────
# COUNTS + SELECTION
# ─────────────────────────────────────────────────

def _dots(n, key="BRPE WBN", color="red") -> list[AerialDot]:
    t = _circle_template()
    out = []
    for i in range(n):
        d = AerialDot(cx=float(i), cy=0, color=color, shape="circle",
                      area=10, template=t, quality=float(i))
        d.class_name = key
        out.append(d)
    return out


class TestCounts:

    def test_class_counts(self):
        assert class_counts(_dots(5)) == {"BRPE WBN": 5}

    def test_class_counts_fallback_to_color_shape(self):
        d = AerialDot(cx=0, cy=0, color="red", shape="circle",
                      area=10, template=_circle_template())
        assert class_counts([d]) == {"red/circle": 1}


class TestSelectByCount:

    def test_keeps_top_n_by_quality(self):
        kept = select_by_count(_dots(10), {"BRPE WBN": 3})
        assert len(kept) == 3
        assert sorted(d.quality for d in kept) == [7.0, 8.0, 9.0]

    def test_zero_count_drops_class(self):
        kept = select_by_count(_dots(5), {"BRPE WBN": 0})
        assert kept == []

    def test_keep_unknown_true_keeps_unlisted(self):
        kept = select_by_count(_dots(4, key="UNLISTED"), {}, keep_unknown=True)
        assert len(kept) == 4

    def test_keep_unknown_false_drops_unlisted(self):
        kept = select_by_count(_dots(4, key="UNLISTED"), {}, keep_unknown=False)
        assert kept == []

    def test_quality_min_filters_unlisted(self):
        kept = select_by_count(
            _dots(10, key="UNLISTED"), {}, keep_unknown=True, quality_min=5.0
        )
        # qualities 0..9; keep those >= 5 -> 5,6,7,8,9
        assert len(kept) == 5


# ─────────────────────────────────────────────────
# REAL IMAGE INTEGRATION
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


class TestRealImagePipeline:

    def test_full_pipeline_runs(self, screenshot):
        """parse_screenshot -> detect_dots(exclude=bbox) -> assign_classes."""
        entries, bbox = parse_screenshot(screenshot)
        dots = detect_dots(screenshot, exclude=bbox)
        assign_classes(dots, entries)
        assert len(dots) > 0

    def _in_box(self, dots, bbox) -> int:
        x, y, w, h = bbox
        return sum(x <= d.cx <= x + w and y <= d.cy <= y + h for d in dots)

    def test_legend_markers_excluded(self, screenshot):
        """Excluding the dialog bbox sharply drops in-box detections (legend
        markers), so they aren't counted as aerial dots."""
        _, bbox = parse_screenshot(screenshot)
        with_excl = self._in_box(detect_dots(screenshot, exclude=bbox), bbox)
        without = self._in_box(detect_dots(screenshot), bbox)
        # Exclusion drops the isolated legend markers; a few split sub-centers
        # from blobs straddling the padded edge may remain.
        assert with_excl < without

    def test_stateless(self, screenshot):
        a = detect_dots(screenshot)
        b = detect_dots(screenshot)
        assert len(a) == len(b)
