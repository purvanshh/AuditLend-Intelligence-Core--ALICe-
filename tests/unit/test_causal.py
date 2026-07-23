"""Tests for causal inference modules (PSM and Synthetic Control)."""

from __future__ import annotations

from math import sqrt
from statistics import mean, stdev

from ml.causal.psm import (
    PSMResult,
    PropensityScoreMatcher,
    compute_balance,
)
from ml.causal.synthetic_control import SyntheticControl


def _make_features(n: int, base: float = 0.0, noise: float = 0.1) -> list[dict[str, float]]:
    import random
    rng = random.Random(42)
    return [
        {"income": base + rng.gauss(0, noise) * (i + 1),
         "age": 30 + rng.gauss(0, noise) * (i + 1),
         "dti": 0.3 + rng.gauss(0, noise * 0.1) * (i + 1)}
        for i in range(n)
    ]


def test_fit_propensity_returns_valid_scores() -> None:
    features = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 3.0}, {"x": 3.0, "y": 4.0},
                {"x": 4.0, "y": 5.0}, {"x": 5.0, "y": 6.0}]
    treatment = [1, 1, 0, 0, 0]
    matcher = PropensityScoreMatcher()
    scores = matcher.fit_propensity(features, treatment)

    assert len(scores) == len(features)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_fit_propensity_fallback_scoring_fn() -> None:
    features = [{"x": 1.0}, {"x": 10.0}]
    treatment = [0, 1]

    matcher = PropensityScoreMatcher()
    scores = matcher.fit_propensity(features, treatment, scoring_fn=lambda f: f["x"] / 10.0)

    assert scores == [0.1, 1.0]


def test_match_produces_balanced_pairs() -> None:
    matcher = PropensityScoreMatcher(caliper=0.5)
    t_feat = [{"income": 50, "age": 35}, {"income": 52, "age": 36}]
    c_feat = [{"income": 49, "age": 34}, {"income": 70, "age": 45}, {"income": 51, "age": 37}]
    t_out = [0.05, 0.08]
    c_out = [0.04, 0.09, 0.06]

    result = matcher.match(t_feat, c_feat, t_out, c_out)

    assert result.n_matched > 0
    assert result.n_matched <= len(t_feat)
    assert len(result.matched_pairs) == result.n_matched


def test_compute_balance_returns_correct_std_diff() -> None:
    t_feat = [{"x": 10.0}, {"x": 12.0}, {"x": 11.0}]
    c_feat = [{"x": 5.0}, {"x": 6.0}, {"x": 7.0}]

    balance = compute_balance(t_feat, c_feat)

    assert "x" in balance
    std_diff = balance["x"]["std_diff"]
    assert std_diff > 0


def test_psm_result_att_calculation() -> None:
    t_out = [0.10, 0.12, 0.08]
    c_out = [0.14, 0.16, 0.10]
    diffs = [t - c for t, c in zip(t_out, c_out)]
    expected_att = mean(diffs)

    result = PSMResult(
        matched_pairs=[],
        treatment_outcomes=t_out,
        control_outcomes=c_out,
        att=expected_att,
        att_ci=(expected_att - 0.01, expected_att + 0.01),
        balance_statistics={},
        n_treatment=3,
        n_control=5,
        n_matched=3,
    )

    assert abs(result.att - expected_att) < 1e-10


def test_empty_input_handling() -> None:
    matcher = PropensityScoreMatcher()
    scores = matcher.fit_propensity([], [])
    assert scores == []

    result = matcher.match([], [], [], [])
    assert result.n_matched == 0
    assert result.att == 0.0


def test_caliper_filtering_works() -> None:
    tight = PropensityScoreMatcher(caliper=0.01)
    wide = PropensityScoreMatcher(caliper=10.0)

    t_feat = [{"z": 1.0}, {"z": 100.0}]
    c_feat = [{"z": 1.1}, {"z": 50.0}, {"z": 99.0}]
    t_out = [0.05, 0.10]
    c_out = [0.04, 0.08, 0.09]

    tight_result = tight.match(t_feat, c_feat, t_out, c_out)
    wide_result = wide.match(t_feat, c_feat, t_out, c_out)

    assert tight_result.n_matched < wide_result.n_matched or tight_result.n_matched == wide_result.n_matched
    assert wide_result.n_matched >= 1


def test_synthetic_control_weights_sum_to_one() -> None:
    sc = SyntheticControl()
    treated_pre = [1.0, 2.0, 3.0]
    control_pool = {
        "A": [1.1, 2.1, 2.9],
        "B": [0.9, 1.8, 3.2],
        "C": [1.2, 2.3, 2.8],
    }

    result = sc.fit(treated_pre, control_pool)

    total_weight = sum(result.weights.values())
    assert abs(total_weight - 1.0) < 1e-4


def test_synthetic_control_weights_non_negative() -> None:
    sc = SyntheticControl()
    treated_pre = [1.0, 2.0, 3.0]
    control_pool = {
        "A": [1.1, 2.1, 2.9],
        "B": [0.9, 1.8, 3.2],
    }

    result = sc.fit(treated_pre, control_pool)

    for w in result.weights.values():
        assert w >= -1e-10


def test_synthetic_control_pretreatment_rmse() -> None:
    sc = SyntheticControl()
    treated_pre = [1.0, 2.0, 3.0]
    control_pool = {
        "A": [1.05, 2.05, 2.95],
    }

    result = sc.fit(treated_pre, control_pool)

    expected_rmse = sqrt(
        mean([(1.0 - 1.05) ** 2, (2.0 - 2.05) ** 2, (3.0 - 2.95) ** 2])
    )
    assert result.pre_treatment_rmse < 0.1
    assert result.pre_treatment_rmse > 0


def test_synthetic_control_post_treatment_effect() -> None:
    sc = SyntheticControl()
    treated_pre = [10.0, 11.0, 12.0]
    control_pool = {
        "A": [9.5, 10.5, 11.5],
        "B": [10.5, 11.5, 12.5],
    }
    treated_post = [13.0, 14.5]
    control_post = {
        "A": [12.0, 12.5],
        "B": [13.5, 14.0],
    }

    result = sc.fit(treated_pre, control_pool, treated_post, control_post)

    assert result.causal_effect != 0.0
    assert len(result.observed_outcomes) == 5
    assert len(result.synthetic_outcomes) == 5


def test_psm_with_custom_ids() -> None:
    matcher = PropensityScoreMatcher(caliper=0.5)
    t_feat = [{"v": 1.0}, {"v": 2.0}]
    c_feat = [{"v": 1.1}, {"v": 2.5}, {"v": 3.0}]
    t_out = [0.1, 0.2]
    c_out = [0.11, 0.22, 0.3]
    t_ids = ["app-1", "app-2"]
    c_ids = ["ctrl-1", "ctrl-2", "ctrl-3"]

    result = matcher.match(t_feat, c_feat, t_out, c_out, t_ids, c_ids)

    for pair in result.matched_pairs:
        assert pair.treatment_id.startswith("app-")
        assert pair.control_id.startswith("ctrl-")


def test_empty_compute_balance() -> None:
    assert compute_balance([], []) == {}
