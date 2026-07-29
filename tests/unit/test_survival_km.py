"""Tests for Kaplan-Meier survival analysis (ml/models/survival_km.py)."""

from __future__ import annotations

import pytest

from ml.models.survival_km import (
    KaplanMeierFitter,
    KaplanMeierResult,
    SurvivalCurve,
    _z_score,
)


# ---------------------------------------------------------------------------
# SurvivalCurve helpers
# ---------------------------------------------------------------------------


def _make_curve(times, probs, lower=None, upper=None) -> SurvivalCurve:
    return SurvivalCurve(
        times=times,
        survival_probabilities=probs,
        confidence_intervals_lower=lower,
        confidence_intervals_upper=upper,
    )


class TestSurvivalCurve:
    def test_survival_at_exact_time(self):
        curve = _make_curve([1.0, 2.0, 3.0], [0.9, 0.75, 0.6])
        assert curve.survival_at_time(1.0) == pytest.approx(0.9)
        assert curve.survival_at_time(2.0) == pytest.approx(0.75)
        assert curve.survival_at_time(3.0) == pytest.approx(0.6)

    def test_survival_before_first_event_returns_first(self):
        curve = _make_curve([5.0, 10.0], [0.8, 0.6])
        assert curve.survival_at_time(3.0) == pytest.approx(0.8)

    def test_survival_after_last_event_returns_last(self):
        curve = _make_curve([1.0, 2.0], [0.9, 0.7])
        assert curve.survival_at_time(99.0) == pytest.approx(0.7)

    def test_survival_empty_returns_1(self):
        curve = _make_curve([], [])
        assert curve.survival_at_time(5.0) == 1.0

    def test_median_survival_time_found(self):
        curve = _make_curve([1.0, 2.0, 3.0], [0.9, 0.6, 0.4])
        assert curve.median_survival_time() == pytest.approx(3.0)

    def test_median_survival_time_first_step(self):
        curve = _make_curve([1.0, 2.0], [0.45, 0.2])
        assert curve.median_survival_time() == pytest.approx(1.0)

    def test_median_survival_time_none_when_no_crossing(self):
        curve = _make_curve([1.0, 2.0], [0.9, 0.7])
        assert curve.median_survival_time() is None

    def test_hazard_at_time_first_step(self):
        curve = _make_curve([1.0, 2.0], [0.8, 0.6])
        # At the first event time, hazard = 1 - S(0)
        h = curve.hazard_at_time(1.0)
        assert h == pytest.approx(1.0 - 0.8)

    def test_hazard_at_time_subsequent_step(self):
        curve = _make_curve([1.0, 2.0], [0.8, 0.6])
        # h(2) = 1 - S(2)/S(1) = 1 - 0.6/0.8
        h = curve.hazard_at_time(2.0)
        assert h == pytest.approx(1.0 - 0.6 / 0.8)

    def test_hazard_after_last_time_is_zero(self):
        curve = _make_curve([1.0, 2.0], [0.8, 0.6])
        assert curve.hazard_at_time(100.0) == 0.0

    def test_hazard_zero_survival_returns_zero(self):
        # Survival drops to 0 — hazard of subsequent step should return 0
        curve = _make_curve([1.0, 2.0], [0.0, 0.0])
        assert curve.hazard_at_time(2.0) == 0.0


# ---------------------------------------------------------------------------
# KaplanMeierFitter.fit
# ---------------------------------------------------------------------------


class TestKaplanMeierFitterFit:
    def test_empty_input_returns_empty_result(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit([], [])
        assert isinstance(result, KaplanMeierResult)
        assert result.overall_curve is None or result.strata == {}

    def test_single_event_survival(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit([3.0], [True])
        assert result.overall_curve is not None
        curve = result.overall_curve
        assert len(curve.times) == 1
        assert curve.survival_probabilities[0] < 1.0

    def test_all_censored_no_events(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit([1.0, 2.0, 3.0], [False, False, False])
        # No events: survival stays at 1.0 throughout
        curve = result.overall_curve
        assert all(s == pytest.approx(1.0) for s in curve.survival_probabilities)

    def test_survival_monotone_decreasing(self):
        fitter = KaplanMeierFitter()
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        events = [1, 1, 1, 1, 1]
        result = fitter.fit(durations, events)
        probs = result.overall_curve.survival_probabilities
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i - 1] + 1e-9

    def test_confidence_intervals_present(self):
        fitter = KaplanMeierFitter()
        durations = [1.0, 2.0, 3.0, 4.0]
        events = [1, 1, 0, 1]
        result = fitter.fit(durations, events)
        curve = result.overall_curve
        assert curve.confidence_intervals_lower is not None
        assert curve.confidence_intervals_upper is not None
        # CI arrays are computed per event; they may be shorter than times
        # (censored time-steps carry forward the last variance term).
        assert len(curve.confidence_intervals_lower) >= 1

    def test_confidence_intervals_bounds(self):
        fitter = KaplanMeierFitter()
        durations = [1.0, 2.0, 3.0, 4.0]
        events = [1, 1, 0, 1]
        result = fitter.fit(durations, events)
        curve = result.overall_curve
        for lo, hi in zip(curve.confidence_intervals_lower, curve.confidence_intervals_upper):
            assert lo >= 0.0
            assert hi <= 1.0
            assert lo <= hi + 1e-9

    def test_strata_counts_populated(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit([1.0, 2.0, 3.0], [1, 0, 1])
        assert result.strata_counts.get("all") == 3
        assert result.strata_event_counts.get("all") == 2

    def test_tied_event_times_handled(self):
        fitter = KaplanMeierFitter()
        # Two events at the same time
        result = fitter.fit([2.0, 2.0, 4.0], [1, 1, 0])
        probs = result.overall_curve.survival_probabilities
        assert len(probs) >= 1

    def test_censored_before_event(self):
        fitter = KaplanMeierFitter()
        # censored at t=1, event at t=5: only 1 subject at risk at t=5
        result = fitter.fit([1.0, 5.0], [False, True])
        # One event observed: survival should drop
        probs = result.overall_curve.survival_probabilities
        assert probs[-1] < 1.0

    def test_large_dataset(self):
        fitter = KaplanMeierFitter()
        durations = list(range(1, 51))
        events = [i % 3 == 0 for i in range(1, 51)]
        result = fitter.fit(durations, events)
        assert result.overall_curve is not None
        assert result.overall_curve.survival_probabilities[-1] >= 0.0


# ---------------------------------------------------------------------------
# KaplanMeierFitter.fit_stratified
# ---------------------------------------------------------------------------


class TestKaplanMeierFitterStratified:
    def _make_rows(self):
        return [
            {"duration_months": 6, "target_defaulted": 1, "grade": "A"},
            {"duration_months": 12, "target_defaulted": 0, "grade": "A"},
            {"duration_months": 4, "target_defaulted": 1, "grade": "B"},
            {"duration_months": 8, "target_defaulted": 1, "grade": "B"},
            {"duration_months": 3, "target_defaulted": 0, "grade": "A"},
        ]

    def test_returns_strata_keys(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(self._make_rows())
        assert "A" in result.strata
        assert "B" in result.strata

    def test_strata_counts_correct(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(self._make_rows())
        assert result.strata_counts["A"] == 3
        assert result.strata_counts["B"] == 2

    def test_strata_event_counts_correct(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(self._make_rows())
        assert result.strata_event_counts["A"] == 1
        assert result.strata_event_counts["B"] == 2

    def test_overall_curve_is_none_in_stratified(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(self._make_rows())
        assert result.overall_curve is None

    def test_empty_rows_returns_empty_strata(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified([])
        assert result.strata == {}

    def test_custom_strata_col(self):
        rows = [
            {"duration_months": 6, "target_defaulted": 1, "purpose": "personal"},
            {"duration_months": 12, "target_defaulted": 0, "purpose": "business"},
        ]
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(rows, strata_col="purpose")
        assert "personal" in result.strata
        assert "business" in result.strata

    def test_missing_strata_col_uses_unknown(self):
        rows = [
            {"duration_months": 6, "target_defaulted": 1},
        ]
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(rows, strata_col="grade")
        assert "UNKNOWN" in result.strata

    def test_each_stratum_survival_is_valid_curve(self):
        fitter = KaplanMeierFitter()
        result = fitter.fit_stratified(self._make_rows())
        for name, curve in result.strata.items():
            assert isinstance(curve, SurvivalCurve)
            for p in curve.survival_probabilities:
                assert 0.0 <= p <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# _z_score
# ---------------------------------------------------------------------------


class TestZScore:
    def test_99_confidence(self):
        assert _z_score(0.99) == pytest.approx(2.576)

    def test_95_confidence(self):
        assert _z_score(0.95) == pytest.approx(1.96)

    def test_90_confidence(self):
        assert _z_score(0.90) == pytest.approx(1.645)

    def test_other_defaults_to_196(self):
        assert _z_score(0.80) == pytest.approx(1.96)
