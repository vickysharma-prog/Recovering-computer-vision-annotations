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
    _lightness_offset,
    _color_candidates,
    _uncounted_capacity,
    _row_offsets,
    _pair_score,
    _SCORE_COLOR,
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
        # A perfect template match is NCC 1.0, and the dot sits exactly on the row's
        # colour, so it also collects the full `_SCORE_COLOR` term.
        assert d.match_score == pytest.approx(1.0 + _SCORE_COLOR)

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


# ─────────────────────────────────────────────────
# PER-FRAME LIGHTNESS OFFSET
# ─────────────────────────────────────────────────

class TestLightnessOffset:
    """A legend glyph is crisp on a white cell; the same marker in the aerial is a
    thin stroke over vegetation at a quarter of the resolution, so background bleeds
    into it and its lightness collapses while its hue survives. On
    `19May18Camera2-Card1-00620` that put 152 of 183 detections beyond the reject
    distance before they could be classified at all.
    """

    @staticmethod
    def _legend(*lab_vecs):
        pal = []
        for i, v in enumerate(lab_vecs):
            e = LegendEntry(row=i, cy=float(i), cx=0.0, shape="circle", color="yellow",
                            hue=30.0, marker=np.zeros((2, 2, 3), np.uint8),
                            template=np.zeros((24, 24), np.float32),
                            class_name=f"C{i}")
            pal.append((e, np.array(v, np.float32)))
        return pal

    @staticmethod
    def _dots(*lab_vecs):
        return [AerialDot(cx=float(i), cy=0.0, color="yellow", shape="circle", area=10,
                          template=np.zeros((24, 24), np.float32),
                          color_vec=np.array(v, np.float32))
                for i, v in enumerate(lab_vecs)]

    def test_uniformly_darker_dots_give_a_negative_offset(self):
        pal = self._legend([240.0, 110.0, 190.0])
        dots = self._dots(*([[160.0, 110.0, 190.0]] * 12))
        assert _lightness_offset(dots, pal) == pytest.approx(-80.0)

    def test_offset_ignores_a_minority_on_odd_backgrounds(self):
        # The median, not the mean: a few dots over unusual ground must not drag it.
        pal = self._legend([240.0, 110.0, 190.0])
        vecs = [[160.0, 110.0, 190.0]] * 10 + [[40.0, 110.0, 190.0]] * 2
        assert _lightness_offset(self._dots(*vecs), pal) == pytest.approx(-80.0)

    def test_too_few_dots_means_no_correction(self):
        # Below the minimum the median is not evidence, so behaviour is unchanged.
        pal = self._legend([240.0, 110.0, 190.0])
        assert _lightness_offset(self._dots([160.0, 110.0, 190.0]), pal) == 0.0


    def test_off_palette_colours_are_not_evidence(self):
        # A dot of a colour the legend does not contain says nothing about the
        # frame's lightness, so it must not enter the median.
        pal = self._legend([240.0, 110.0, 190.0])
        dots = self._dots(*([[160.0, 200.0, 60.0]] * 12))
        assert _lightness_offset(dots, pal) == 0.0

    def test_offset_rescues_a_dot_the_reject_would_drop(self):
        pal = self._legend([240.0, 110.0, 190.0])
        dark = self._dots([160.0, 110.0, 190.0])[0]
        assert _color_candidates(dark, pal, 0.0) == []
        assert len(_color_candidates(dark, pal, -80.0)) == 1

    def test_offset_does_not_admit_a_wrong_colour(self):
        # The shift moves lightness only, so a dot whose chroma is off-palette stays
        # rejected — the reject still rejects.
        pal = self._legend([240.0, 110.0, 190.0])
        wrong = self._dots([160.0, 200.0, 60.0])[0]
        assert _color_candidates(wrong, pal, -80.0) == []

    def test_between_class_lightness_survives_the_shift(self):
        # Two classes separated only by lightness must stay separable, which is why
        # the correction is a shift and not a discard of L.
        pal = self._legend([240.0, 110.0, 190.0], [180.0, 110.0, 190.0])
        light = self._dots([200.0, 110.0, 190.0])[0]
        cands = _color_candidates(light, pal, -40.0)
        assert [e.class_name for e in cands] == ["C0"]


class TestUncountedCapacity:
    """
    A row whose Count never read used to be unlimited, so a row parsed off the
    bottom of the table — the scrollbar strip, or the photo below the dialog — could
    absorb dots belonging to real rows. On `17May10Camera2-Card1-5745` two such rows
    took 49 detections, 31 of them real `LAGU sit`, while row 0 filled to 81 of its
    stated 150.

    The rule has to be narrow. Zeroing *every* unread row also destroys frames whose
    Count column simply failed to OCR: on `14June21Camera1-Card1-238`, where 3 of 13
    rows read a count, per-dot accuracy fell 0.667 → 0.018. Only rows below the last
    counted row are outside the table.
    """

    @staticmethod
    def _legend(*counts):
        return [LegendEntry(row=i, cy=float(i * 20), cx=0.0, shape="circle",
                            color="red", hue=0.0,
                            marker=np.zeros((2, 2, 3), np.uint8),
                            template=np.zeros((24, 24), np.float32), count=c)
                for i, c in enumerate(counts)]

    def test_trailing_unread_rows_are_zeroed(self, monkeypatch):
        monkeypatch.setattr("src.classify._UNCOUNTED", "tail")
        legend = self._legend(5, 3, None, None)          # 5745's shape
        assert _uncounted_capacity(legend) == {id(legend[2]): 0, id(legend[3]): 0}

    def test_an_unread_row_between_counted_rows_is_left_alone(self, monkeypatch):
        # A genuine row whose digits failed to OCR must keep its old freedom.
        monkeypatch.setattr("src.classify._UNCOUNTED", "tail")
        assert _uncounted_capacity(self._legend(5, None, 3)) == {}

    def test_no_counts_at_all_changes_nothing(self, monkeypatch):
        # With nothing to anchor "below", the rule has no opinion.
        monkeypatch.setattr("src.classify._UNCOUNTED", "tail")
        assert _uncounted_capacity(self._legend(None, None)) == {}

    def test_zero_caps_every_unread_row(self, monkeypatch):
        monkeypatch.setattr("src.classify._UNCOUNTED", "zero")
        legend = self._legend(5, None, 3, None)
        assert _uncounted_capacity(legend) == {id(legend[1]): 0, id(legend[3]): 0}

    def test_open_restores_the_original_behaviour(self, monkeypatch):
        monkeypatch.setattr("src.classify._UNCOUNTED", "open")
        assert _uncounted_capacity(self._legend(5, None, None)) == {}

    def test_a_fully_counted_legend_is_untouched(self, monkeypatch):
        monkeypatch.setattr("src.classify._UNCOUNTED", "tail")
        assert _uncounted_capacity(self._legend(5, 3, 1)) == {}


class TestRowOffsets:
    """
    The legend-to-aerial colour drift is **per row**, not per frame.

    `_lightness_offset` applies one shift to the whole frame. Against the hand
    labels that is the wrong shape: on `17May10Camera2-Card1-5745` the frame-wide
    median is `a = -12.5` while row 0 needs `-29.4` and row 2 `-11.0`, and on
    `19May18Camera2-Card1-00620` row 0 needs `L = -84.3` against row 1's `-16.0`.
    The cause is the glyph, not the class — a thin asterisk dissolves into the
    background where a filled circle keeps its colour.

    A second pass estimates each row's own drift from the dots the first pass gave
    it, and those offsets only ever **add** candidates.
    """

    @staticmethod
    def _palette(*vecs):
        out = []
        for i, v in enumerate(vecs):
            e = LegendEntry(row=i, cy=float(i), cx=0.0, shape="circle", color="red",
                            hue=0.0, marker=np.zeros((2, 2, 3), np.uint8),
                            template=np.zeros((24, 24), np.float32),
                            class_name=f"C{i}")
            out.append((e, np.array(v, np.float32)))
        return out

    @staticmethod
    def _dot(vec, row):
        d = AerialDot(cx=0.0, cy=0.0, color="red", shape="circle", area=10,
                      template=np.zeros((24, 24), np.float32),
                      color_vec=np.array(vec, np.float32))
        d.legend_row = row
        return d

    def test_each_row_gets_its_own_offset(self):
        pal = self._palette([240.0, 180.0, 120.0], [240.0, 180.0, 120.0])
        dots = ([self._dot([200.0, 180.0, 120.0], 0)] * 8 +
                [self._dot([100.0, 180.0, 120.0], 1)] * 8)
        off = _row_offsets(dots, pal)
        assert off[0][0] == pytest.approx(-40.0)
        assert off[1][0] == pytest.approx(-140.0)

    def test_all_three_axes_are_corrected(self):
        # The frame-wide shift moves L only; the measured drift moves chroma too.
        pal = self._palette([240.0, 180.0, 120.0])
        dots = [self._dot([230.0, 150.0, 135.0], 0)] * 8
        assert list(_row_offsets(dots, pal)[0]) == pytest.approx([-10.0, -30.0, 15.0])

    def test_a_row_with_too_few_dots_is_skipped(self):
        # Below the minimum a median is not evidence, so the row keeps the
        # frame-wide shift rather than a noisy one of its own.
        pal = self._palette([240.0, 180.0, 120.0])
        assert _row_offsets([self._dot([200.0, 180.0, 120.0], 0)] * 3, pal) == {}

    def test_unassigned_dots_contribute_nothing(self):
        pal = self._palette([240.0, 180.0, 120.0])
        assert _row_offsets([self._dot([200.0, 180.0, 120.0], None)] * 8, pal) == {}

    def test_offsets_only_add_candidates_never_remove(self):
        # Containment: a row the frame-wide shift already found must survive
        # whatever the per-row estimates say, because a contaminated estimate
        # (measured at 76.7 away on one row of `00620`) must not be able to
        # discard a good candidate.
        pal = self._palette([240.0, 180.0, 120.0], [100.0, 180.0, 120.0])
        d = self._dot([238.0, 180.0, 120.0], None)
        before = _color_candidates(d, pal, 0.0)
        after = _color_candidates(d, pal, 0.0, {1: np.array([-500.0, 0.0, 0.0],
                                                            np.float32)})
        assert {e.row for e in before} <= {e.row for e in after}

    def test_a_rows_own_offset_can_win_it_a_dot(self):
        # The point of the second pass: a dot whose own row sat outside the
        # frame-wide margin becomes reachable under that row's measured drift.
        pal = self._palette([240.0, 180.0, 120.0], [240.0, 60.0, 120.0])
        d = self._dot([240.0, 150.0, 120.0], None)       # 30 from row 0, 90 from row 1
        assert {e.row for e in _color_candidates(d, pal, 0.0)} == {0}
        cands = _color_candidates(d, pal, 0.0, {1: np.array([0.0, 85.0, 0.0],
                                                            np.float32)})
        assert {e.row for e in cands} == {0, 1}

    def test_disabled_returns_nothing(self, monkeypatch):
        monkeypatch.setattr("src.classify._ROW_OFFSET", False)
        pal = self._palette([240.0, 180.0, 120.0])
        assert _row_offsets([self._dot([200.0, 180.0, 120.0], 0)] * 8, pal) == {}


class TestColourInThePairScore:
    """
    Colour used only to gate candidacy, never to rank.

    A dot sitting exactly on a row's colour and one that scraped in at the edge of
    the margin therefore competed as equals for the same slot, and the template NCC
    alone decided. On `5745` that left 44 real markers at a full row. Folding the
    per-row colour distance into the score moved the four labelled frames from 0.789
    to 0.828 pooled.
    """

    @staticmethod
    def _pair():
        e = LegendEntry(row=0, cy=0.0, cx=0.0, shape="circle", color="red", hue=0.0,
                        marker=np.zeros((2, 2, 3), np.uint8),
                        template=np.zeros((24, 24), np.float32))
        d = AerialDot(cx=0.0, cy=0.0, color="red", shape="circle", area=10,
                      template=np.zeros((24, 24), np.float32))
        return d, e

    def test_a_closer_colour_scores_higher(self, monkeypatch):
        monkeypatch.setattr("src.classify._SCORE_COLOR", 0.2)
        d, e = self._pair()
        assert _pair_score(d, e, False, 2.0) > _pair_score(d, e, False, 40.0)

    def test_colour_beyond_the_reject_adds_nothing(self, monkeypatch):
        # Clamped at zero, so a far colour cannot drag a good template match below
        # a pair that carries no colour information at all.
        monkeypatch.setattr("src.classify._SCORE_COLOR", 0.2)
        d, e = self._pair()
        assert _pair_score(d, e, False, 999.0) == _pair_score(d, e, False, None)

    def test_no_colour_information_leaves_the_score_alone(self, monkeypatch):
        monkeypatch.setattr("src.classify._SCORE_COLOR", 0.2)
        d, e = self._pair()
        assert _pair_score(d, e, False, None) == _pair_score(d, e, False)

    def test_weight_zero_restores_the_old_score(self, monkeypatch):
        monkeypatch.setattr("src.classify._SCORE_COLOR", 0.0)
        d, e = self._pair()
        assert _pair_score(d, e, False, 0.0) == _pair_score(d, e, False, 40.0)


class TestBlockedDotRetry:
    """
    A dot whose every candidate row filled up used to be dropped.

    It already falls through to its next-best *candidate*, so this only concerns the
    dot with nowhere left to fall. On `17May10Camera2-Card1-5745` that was 25
    detections whose single candidate is `ROSP bird`, a row the dialog genuinely
    counts as zero — the count is right, the candidate set was simply too narrow to
    also offer `ROSP site` next door.

    Two boundaries matter and both were measured. A dot colour **rejected** outright
    is off-palette, which is the valid/invalid decision working, and must stay
    unassigned. And the retry runs only in the final pass: letting it place dots in
    the first pass changed which dots each row held, moved the per-row offsets, and
    cost `00620` five dots.
    """

    @staticmethod
    def _legend(*specs):
        out = []
        for i, (vec, count) in enumerate(specs):
            e = LegendEntry(row=i, cy=0.0, cx=0.0, shape="circle", color="red",
                            hue=0.0, marker=np.zeros((2, 2, 3), np.uint8),
                            template=_circle_template(), class_name=f"C{i}",
                            count=count)
            e._color_vec = np.array(vec, np.float32)
            e._shape_tmpl = _circle_template()
            out.append(e)
        return out

    @staticmethod
    def _dots(*vecs):
        return [AerialDot(cx=float(i), cy=0.0, color="red", shape="circle", area=10,
                          template=_circle_template(),
                          color_vec=np.array(v, np.float32))
                for i, v in enumerate(vecs)]

    def test_a_dot_blocked_by_a_zero_count_row_finds_the_row_next_door(self):
        # `5745` in miniature: the dot's colour picks only the zero-count row, and
        # the row it belongs to sits just outside the margin with space free.
        legend = self._legend(([130.0, 185.0, 170.0], 0),      # full at zero
                              ([130.0, 150.0, 170.0], 5))      # room, further away
        d = self._dots([130.0, 184.0, 170.0])[0]
        assign_classes([d], legend)
        assert d.legend_row == 1

    def test_an_off_palette_dot_stays_unassigned(self, monkeypatch):
        # Colour rejected it outright, so it is noise as far as the legend knows.
        # Forcing it onto a row is what the retry must not do.
        legend = self._legend(([130.0, 185.0, 170.0], 0))
        d = self._dots([40.0, 40.0, 40.0])[0]
        assign_classes([d], legend)
        assert d.legend_row is None

    def test_a_dot_that_already_has_a_row_is_untouched(self):
        legend = self._legend(([130.0, 185.0, 170.0], 5),
                              ([130.0, 150.0, 170.0], 5))
        d = self._dots([130.0, 184.0, 170.0])[0]
        assign_classes([d], legend)
        assert d.legend_row == 0

    def test_disabled_leaves_the_dot_unassigned(self, monkeypatch):
        monkeypatch.setattr("src.classify._BLOCKED_RETRY", False)
        legend = self._legend(([130.0, 185.0, 170.0], 0),
                              ([130.0, 150.0, 170.0], 5))
        d = self._dots([130.0, 184.0, 170.0])[0]
        assign_classes([d], legend)
        assert d.legend_row is None
