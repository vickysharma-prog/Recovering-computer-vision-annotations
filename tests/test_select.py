"""Tests for the frame-selection bound (src/select.py)."""

import pytest

from src.select import DEFAULT_QUALITY, accept_frame, frame_ratio


class TestFrameRatio:

    def test_plain_ratio(self):
        assert frame_ratio(120, 100) == pytest.approx(1.2)

    def test_missing_count_has_no_bound(self):
        # No count means no ceiling can be formed, which is not the same as a
        # frame that holds no birds.
        assert frame_ratio(50, None) is None
        assert frame_ratio(50, 0) is None
        assert frame_ratio(50, -3) is None


class TestAcceptFrame:

    def test_exact_match_accepted(self):
        d = accept_frame(100, 100)
        assert d.accepted
        assert d.ratio == pytest.approx(1.0)
        assert d.precision_ceiling == pytest.approx(1.0)
        assert d.recall_ceiling == pytest.approx(1.0)

    def test_over_detection_rejected_on_precision(self):
        # 7x over-detection caps precision at 1/7 = 0.14 whatever the detector does.
        d = accept_frame(700, 100)
        assert not d.accepted
        assert d.precision_ceiling == pytest.approx(1 / 7)
        assert "precision" in d.reason

    def test_under_detection_rejected_on_recall(self):
        # The one-sided version of this rule accepted 19May18Camera1-Card1-00622,
        # which returns a single detection against 19 dots. Recall is capped at
        # 0.05 there, so the two-sided band is what rejects it.
        d = accept_frame(1, 19)
        assert not d.accepted
        assert d.recall_ceiling == pytest.approx(1 / 19)
        assert "recall" in d.reason

    def test_band_is_symmetric_in_the_quality_target(self):
        q = 0.6
        # Ratio q and ratio 1/q are both exactly on the boundary, and both are in.
        assert accept_frame(60, 100, quality=q).accepted
        assert accept_frame(int(100 / q), 100, quality=q).accepted
        # Just outside either end is out.
        assert not accept_frame(59, 100, quality=q).accepted
        assert not accept_frame(168, 100, quality=q).accepted

    def test_ceilings_are_at_least_quality_for_anything_accepted(self):
        q = DEFAULT_QUALITY
        for detected in range(1, 400):
            d = accept_frame(detected, 100, quality=q)
            if d.accepted:
                assert d.precision_ceiling >= q - 1e-9
                assert d.recall_ceiling >= q - 1e-9

    def test_nothing_detected_rejected(self):
        d = accept_frame(0, 40)
        assert not d.accepted
        assert d.recall_ceiling == 0.0

    def test_no_count_rejected_rather_than_assumed(self):
        d = accept_frame(120, None)
        assert not d.accepted
        assert d.reason == "no usable count"

    def test_invalid_quality_rejected(self):
        for bad in (0, -0.5, 1.5):
            with pytest.raises(ValueError):
                accept_frame(10, 10, quality=bad)
