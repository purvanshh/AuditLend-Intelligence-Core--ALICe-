"""Tests for champion/challenger comparison (ml/causal/champion_challenger.py)."""

from __future__ import annotations

import pytest

from ml.causal.ab_framework import ExperimentConfig, ExperimentResult
from ml.causal.champion_challenger import ChampionChallengerDecision, compare_arms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    control_size: int = 500,
    treatment_size: int = 500,
    profit_delta: float = 0.0,
    ci: tuple[float, float] = (-0.01, 0.01),
    p_value: float = 0.5,
    significant: bool = False,
    min_sample_size: int = 100,
) -> ExperimentResult:
    config = ExperimentConfig(
        name="test",
        control_arm="heuristic",
        treatment_arm="ml",
        min_sample_size=min_sample_size,
    )
    return ExperimentResult(
        config=config,
        control_results={"profit": 0.05},
        treatment_results={"profit": 0.05 + profit_delta},
        deltas={"profit": profit_delta},
        confidence_intervals={"profit": ci},
        p_values={"profit": p_value},
        significant={"profit": significant},
        sample_size={"heuristic": control_size, "ml": treatment_size},
    )


# ---------------------------------------------------------------------------
# compare_arms — insufficient sample size path
# ---------------------------------------------------------------------------


class TestCompareArmsInsufficientSample:
    def test_returns_tie_when_control_insufficient(self):
        result = _make_result(control_size=50, treatment_size=500, min_sample_size=100)
        decision = compare_arms(result)
        assert decision.winner == "tie"
        assert "CONTINUE" in decision.recommendation
        assert decision.significant is False

    def test_returns_tie_when_treatment_insufficient(self):
        result = _make_result(control_size=500, treatment_size=50, min_sample_size=100)
        decision = compare_arms(result)
        assert decision.winner == "tie"

    def test_total_sample_size_correct(self):
        result = _make_result(control_size=30, treatment_size=40, min_sample_size=100)
        decision = compare_arms(result)
        assert decision.sample_size == 70

    def test_profit_lift_preserved(self):
        result = _make_result(
            control_size=50, treatment_size=50, min_sample_size=200, profit_delta=0.03
        )
        decision = compare_arms(result)
        assert decision.profit_lift == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# compare_arms — challenger wins (PROMOTE)
# ---------------------------------------------------------------------------


class TestCompareArmsPromote:
    def test_promotes_when_significant_positive(self):
        result = _make_result(
            profit_delta=0.05,
            ci=(0.01, 0.09),
            p_value=0.01,
            significant=True,
        )
        decision = compare_arms(result)
        assert decision.winner == "challenger"
        assert "PROMOTE" in decision.recommendation
        assert decision.significant is True

    def test_profit_ci_and_p_preserved(self):
        result = _make_result(
            profit_delta=0.05,
            ci=(0.02, 0.08),
            p_value=0.02,
            significant=True,
        )
        decision = compare_arms(result)
        assert decision.profit_ci == (0.02, 0.08)
        assert decision.p_value == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# compare_arms — champion wins (ROLLBACK)
# ---------------------------------------------------------------------------


class TestCompareArmsRollback:
    def test_rolls_back_when_significant_negative(self):
        result = _make_result(
            profit_delta=-0.04,
            ci=(-0.08, -0.01),
            p_value=0.02,
            significant=True,
        )
        decision = compare_arms(result)
        assert decision.winner == "champion"
        assert "ROLLBACK" in decision.recommendation

    def test_profit_lift_is_negative(self):
        result = _make_result(
            profit_delta=-0.03,
            ci=(-0.06, -0.005),
            p_value=0.03,
            significant=True,
        )
        decision = compare_arms(result)
        assert decision.profit_lift < 0


# ---------------------------------------------------------------------------
# compare_arms — no significant difference (CONTINUE)
# ---------------------------------------------------------------------------


class TestCompareArmsContinue:
    def test_continue_when_not_significant(self):
        result = _make_result(
            profit_delta=0.01,
            ci=(-0.02, 0.04),
            p_value=0.3,
            significant=False,
        )
        decision = compare_arms(result)
        assert decision.winner == "tie"
        assert "CONTINUE" in decision.recommendation
        assert decision.significant is False

    def test_continue_when_ci_crosses_zero(self):
        result = _make_result(
            profit_delta=0.02,
            ci=(-0.01, 0.05),
            p_value=0.08,
            significant=True,  # significant but CI crosses zero
        )
        decision = compare_arms(result)
        assert decision.winner == "tie"
        assert "CONTINUE" in decision.recommendation


# ---------------------------------------------------------------------------
# ChampionChallengerDecision dataclass
# ---------------------------------------------------------------------------


class TestChampionChallengerDecision:
    def test_fields_accessible(self):
        d = ChampionChallengerDecision(
            winner="challenger",
            profit_lift=0.05,
            profit_ci=(0.01, 0.09),
            p_value=0.02,
            significant=True,
            sample_size=1000,
            recommendation="PROMOTE: test",
        )
        assert d.winner == "challenger"
        assert d.profit_lift == pytest.approx(0.05)
        assert d.sample_size == 1000
