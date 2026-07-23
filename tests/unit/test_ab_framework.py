"""Tests for the A/B testing statistical framework."""

from __future__ import annotations

import math
import random

from ml.causal.ab_framework import (
    AB_BUCKET_COUNT,
    PROFIT_PER_LOAN,
    LOSS_GIVEN_DEFAULT,
    ExperimentConfig,
    ExperimentFramework,
)
from ml.causal.champion_challenger import compare_arms


def _make_record(
    arm: str,
    decision: str = "APPROVE",
    defaulted: int = 0,
    loan_amount: float = 10_000.0,
    confidence: float = 0.85,
) -> dict:
    return {
        "arm": arm,
        "decision": decision,
        "defaulted": defaulted,
        "loan_amount": loan_amount,
        "confidence": confidence,
    }


def _profit(loan_amount: float, defaulted: int) -> float:
    return -loan_amount * LOSS_GIVEN_DEFAULT if defaulted else loan_amount * PROFIT_PER_LOAN


# --- Assignment Tests ---

def test_assign_returns_deterministic() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    arm1 = framework.assign("app-1", "A", "debt_consolidation")
    arm2 = framework.assign("app-1", "A", "debt_consolidation")
    assert arm1 == arm2


def test_assign_different_ids_different_strata() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    arm_a = framework.assign("app-1", "A", "debt_consolidation")
    arm_b = framework.assign("app-2", "B", "credit_card")
    # Different strata should be independent
    assert isinstance(arm_a, str)
    assert isinstance(arm_b, str)


def test_assign_within_same_stratum_can_vary_by_id() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    arms = [framework.assign(f"app-{i}", "A", "debt_consolidation") for i in range(100)]
    # Different application IDs within same stratum can be assigned to different arms
    assert all(arm in ("heuristic", "ml") for arm in arms)


def test_assign_splits_balanced_approximately() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    grades = ["A", "B", "C", "D", "E"]
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "other"]
    assignments: list[str] = []
    for grade in grades:
        for purpose in purposes:
            arm = framework.assign(f"app-{grade}-{purpose}", grade, purpose)
            assignments.append(arm)

    control_count = sum(1 for a in assignments if a == config.control_arm)
    treatment_count = sum(1 for a in assignments if a == config.treatment_arm)
    total = len(assignments)
    # Should be roughly balanced (within 80/20 split)
    assert control_count > 0
    assert treatment_count > 0
    # No more than 80% one side
    assert control_count / total <= 0.8
    assert treatment_count / total <= 0.8


def test_assign_without_stratification() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1, stratified=False)
    framework = ExperimentFramework(config)
    arm1 = framework.assign("app-1", "A", "debt_consolidation")
    arm2 = framework.assign("app-1", "A", "debt_consolidation")
    assert arm1 == arm2


def test_assign_uses_correct_arm_names() -> None:
    config = ExperimentConfig(name="test", control_arm="control", treatment_arm="treatment", min_sample_size=1)
    framework = ExperimentFramework(config)
    arm = framework.assign("app-1", "A", "debt_consolidation")
    assert arm in ("control", "treatment")


# --- Analysis Tests ---

def test_analyze_returns_expected_fields() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [_make_record("heuristic"), _make_record("ml")]
    result = framework.analyze(records)

    assert result.config is config
    assert "profit" in result.control_results
    assert "profit" in result.treatment_results
    assert "profit" in result.deltas
    assert "profit" in result.confidence_intervals
    assert "profit" in result.p_values
    assert "profit" in result.significant
    assert "heuristic" in result.sample_size
    assert "ml" in result.sample_size


def test_analyze_with_empty_records() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    result = framework.analyze([])
    assert result.control_results["profit"] == 0.0
    assert result.treatment_results["profit"] == 0.0
    assert result.sample_size["heuristic"] == 0
    assert result.sample_size["ml"] == 0


def test_analyze_profit_no_defaults() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=0, loan_amount=10000.0),
        _make_record("ml", decision="APPROVE", defaulted=0, loan_amount=10000.0),
    ]
    result = framework.analyze(records)
    expected_profit = 10000.0 * PROFIT_PER_LOAN
    assert result.control_results["profit"] == expected_profit
    assert result.treatment_results["profit"] == expected_profit


def test_analyze_profit_with_defaults() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=1, loan_amount=10000.0),
        _make_record("ml", decision="APPROVE", defaulted=1, loan_amount=10000.0),
    ]
    result = framework.analyze(records)
    expected_loss = -10000.0 * LOSS_GIVEN_DEFAULT
    assert result.control_results["profit"] == expected_loss
    assert result.treatment_results["profit"] == expected_loss


def test_analyze_approval_and_default_rates() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=0),
        _make_record("heuristic", decision="APPROVE", defaulted=1),
        _make_record("heuristic", decision="DECLINE", defaulted=0),
        _make_record("heuristic", decision="NEEDS_REVIEW", defaulted=0),
        _make_record("ml", decision="APPROVE", defaulted=0),
        _make_record("ml", decision="APPROVE", defaulted=0),
        _make_record("ml", decision="APPROVE", defaulted=1),
        _make_record("ml", decision="DECLINE", defaulted=0),
    ]
    result = framework.analyze(records)
    assert result.control_results["approval_rate"] == 0.5
    assert result.control_results["default_rate"] == 0.5
    assert result.treatment_results["approval_rate"] == 0.75
    assert abs(result.treatment_results["default_rate"] - 1.0 / 3.0) < 1e-6


def test_analyze_calibration() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", confidence=0.8),
        _make_record("heuristic", confidence=0.9),
        _make_record("ml", confidence=0.85),
        _make_record("ml", confidence=0.95),
    ]
    result = framework.analyze(records)
    assert result.control_results["calibration"] == 0.85
    assert result.treatment_results["calibration"] == 0.9


# --- Bootstrap CI Tests ---

def test_bootstrap_ci_with_known_data() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    control = [100.0] * 100
    treatment = [200.0] * 100
    ci = framework.bootstrap_ci(control, treatment, metric="profit", n_bootstrap=1000)
    lower, upper = ci
    # All treatment values are larger, so delta should be around 100
    assert lower > 0
    assert upper > 0
    assert lower <= upper


def test_bootstrap_ci_reproducible() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    control = [random.uniform(90, 110) for _ in range(50)]
    treatment = [random.uniform(190, 210) for _ in range(50)]
    ci1 = framework.bootstrap_ci(control, treatment, n_bootstrap=1000)
    ci2 = framework.bootstrap_ci(control, treatment, n_bootstrap=1000)
    assert ci1 == ci2


def test_bootstrap_ci_empty_returns_zero() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    assert framework.bootstrap_ci([], []) == (0.0, 0.0)
    assert framework.bootstrap_ci([1.0], []) == (0.0, 0.0)
    assert framework.bootstrap_ci([], [1.0]) == (0.0, 0.0)


def test_bootstrap_ci_contains_true_value() -> None:
    random.seed(12345)
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    true_delta = 50.0
    control = [random.gauss(0, 10) for _ in range(200)]
    treatment = [random.gauss(true_delta, 10) for _ in range(200)]
    ci = framework.bootstrap_ci(control, treatment, n_bootstrap=2000)
    # The true delta should be within the CI (likely, but not guaranteed)
    assert ci[0] <= true_delta <= ci[1], f"True delta {true_delta} not in CI {ci}"


# --- Welch's T-Test Tests ---

def test_welch_ttest_same_distribution() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    values = [random.gauss(100, 15) for _ in range(100)]
    t_stat, p_value = framework.welch_ttest(values, values)
    assert t_stat == 0.0
    assert p_value == 1.0


def test_welch_ttest_different_means() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    control = [100.0 + random.gauss(0, 1) for _ in range(50)]
    treatment = [200.0 + random.gauss(0, 1) for _ in range(50)]
    t_stat, p_value = framework.welch_ttest(control, treatment)
    assert t_stat > 0
    assert p_value < 0.05


def test_welch_ttest_empty_returns_one() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    t_stat, p_value = framework.welch_ttest([], [1.0])
    assert t_stat == 0.0
    assert p_value == 1.0
    t_stat, p_value = framework.welch_ttest([1.0], [])
    assert t_stat == 0.0
    assert p_value == 1.0


def test_welch_ttest_symmetric() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    control = [100.0] * 50
    treatment = [200.0] * 50
    t1, p1 = framework.welch_ttest(control, treatment)
    t2, p2 = framework.welch_ttest(treatment, control)
    assert abs(t1 + t2) < 1e-10
    assert abs(p1 - p2) < 1e-10


def test_welch_ttest_small_sample() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    control = [100.0, 101.0, 99.0, 102.0, 98.0]
    treatment = [200.0, 201.0, 199.0, 202.0, 198.0]
    t_stat, p_value = framework.welch_ttest(control, treatment)
    assert t_stat > 0
    assert p_value < 0.05


# --- Champion/Challenger Decision Tests ---

def test_compare_arms_promotes_challenger() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=1)
    framework = ExperimentFramework(config)
    random.seed(42)
    records = [
        _make_record(
            "heuristic", decision="APPROVE", defaulted=0,
            loan_amount=10000.0 + random.gauss(0, 100),
        )
        for _ in range(100)
    ] + [
        _make_record(
            "ml", decision="APPROVE", defaulted=0,
            loan_amount=20000.0 + random.gauss(0, 100),
        )
        for _ in range(100)
    ]
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert decision.winner == "challenger", f"Expected challenger, got {decision.recommendation}"
    assert decision.significant is True
    assert "PROMOTE" in decision.recommendation


def test_compare_arms_promotes_challenger_lower_defaults() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=1)
    framework = ExperimentFramework(config)
    records = (
        [_make_record("heuristic", decision="APPROVE", defaulted=0, loan_amount=10000.0) for _ in range(50)]
        + [_make_record("heuristic", decision="APPROVE", defaulted=1, loan_amount=10000.0) for _ in range(50)]
        + [_make_record("ml", decision="APPROVE", defaulted=0, loan_amount=10000.0) for _ in range(100)]
    )
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert decision.winner == "challenger"
    assert decision.significant is True
    assert "PROMOTE" in decision.recommendation


def test_compare_arms_rolls_back_challenger() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=1)
    framework = ExperimentFramework(config)
    random.seed(42)
    records = [
        _make_record(
            "ml", decision="APPROVE", defaulted=0,
            loan_amount=10000.0 + random.gauss(0, 100),
        )
        for _ in range(100)
    ] + [
        _make_record(
            "heuristic", decision="APPROVE", defaulted=0,
            loan_amount=20000.0 + random.gauss(0, 100),
        )
        for _ in range(100)
    ]
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert decision.winner == "champion", f"Expected champion, got {decision.recommendation}"
    assert decision.significant is True
    assert "ROLLBACK" in decision.recommendation


def test_compare_arms_continues_when_not_significant() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=0, loan_amount=10000.0)
        for _ in range(50)
    ] + [
        _make_record("ml", decision="APPROVE", defaulted=0, loan_amount=10000.0)
        for _ in range(50)
    ]
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert decision.winner == "tie"
    assert decision.significant is False
    assert "CONTINUE" in decision.recommendation


def test_compare_arms_respects_min_sample_size() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=1000)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=0, loan_amount=10000.0)
        for _ in range(100)
    ] + [
        _make_record("ml", decision="APPROVE", defaulted=0, loan_amount=20000.0)
        for _ in range(100)
    ]
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert decision.winner == "tie"
    assert decision.significant is False
    assert "CONTINUE" in decision.recommendation
    assert "insufficient sample" in decision.recommendation


def test_compare_arms_promote_with_large_enough_sample() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=100)
    framework = ExperimentFramework(config)
    random.seed(42)
    records = [
        _make_record(
            "heuristic", decision="APPROVE", defaulted=0,
            loan_amount=10000.0 + random.gauss(0, 100),
        )
        for _ in range(100)
    ] + [
        _make_record(
            "ml", decision="APPROVE", defaulted=0,
            loan_amount=20000.0 + random.gauss(0, 100),
        )
        for _ in range(100)
    ]
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert "PROMOTE" in decision.recommendation


# --- Config Tests ---

def test_config_defaults() -> None:
    config = ExperimentConfig(name="default_test")
    assert config.control_arm == "heuristic"
    assert config.treatment_arm == "ml"
    assert "profit" in config.metrics
    assert config.alpha == 0.05
    assert config.min_sample_size == 1000
    assert config.stratified is True


def test_config_custom_values() -> None:
    config = ExperimentConfig(
        name="custom",
        control_arm="control",
        treatment_arm="treatment",
        metrics=("profit",),
        alpha=0.01,
        min_sample_size=500,
        stratified=False,
    )
    assert config.control_arm == "control"
    assert config.treatment_arm == "treatment"
    assert config.metrics == ("profit",)
    assert config.alpha == 0.01
    assert config.min_sample_size == 500
    assert config.stratified is False


# --- Edge Cases ---

def test_analyze_only_one_arm_present() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [_make_record("heuristic"), _make_record("heuristic")]
    result = framework.analyze(records)
    assert result.sample_size["heuristic"] == 2
    assert result.sample_size["ml"] == 0
    assert result.deltas["profit"] == -result.control_results["profit"]


def test_analyze_no_approved_loans() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="DECLINE"),
        _make_record("heuristic", decision="NEEDS_REVIEW"),
        _make_record("ml", decision="DECLINE"),
    ]
    result = framework.analyze(records)
    assert result.control_results["approval_rate"] == 0.0
    assert result.control_results["default_rate"] == 0.0
    assert result.control_results["profit"] == 0.0


def test_analyze_with_various_loan_amounts() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=0, loan_amount=5000.0),
        _make_record("heuristic", decision="APPROVE", defaulted=1, loan_amount=15000.0),
        _make_record("ml", decision="APPROVE", defaulted=0, loan_amount=10000.0),
        _make_record("ml", decision="APPROVE", defaulted=0, loan_amount=20000.0),
    ]
    result = framework.analyze(records)
    expected_control_profit = (5000.0 * PROFIT_PER_LOAN + (-15000.0 * LOSS_GIVEN_DEFAULT)) / 2
    expected_treatment_profit = (10000.0 * PROFIT_PER_LOAN + 20000.0 * PROFIT_PER_LOAN) / 2
    assert result.control_results["profit"] == expected_control_profit
    assert result.treatment_results["profit"] == expected_treatment_profit
    assert result.treatment_results["profit"] > result.control_results["profit"]


def test_compare_arms_ci_contains_zero_not_significant() -> None:
    config = ExperimentConfig(name="test", alpha=0.05, min_sample_size=1)
    framework = ExperimentFramework(config)
    # Identical means so CI straddles zero
    random.seed(1234)
    control = [100.0 + random.gauss(0, 20) for _ in range(200)]
    treatment = [100.0 + random.gauss(0, 20) for _ in range(200)]
    records = [
        _make_record("heuristic", decision="APPROVE", defaulted=0, loan_amount=abs(c))
        for c in control
    ] + [
        _make_record("ml", decision="APPROVE", defaulted=0, loan_amount=abs(t))
        for t in treatment
    ]
    result = framework.analyze(records)
    decision = compare_arms(result)
    assert decision.winner == "tie"
    assert decision.significant is False
    assert "CONTINUE" in decision.recommendation


def test_no_arm_field_uses_unknown_arm() -> None:
    config = ExperimentConfig(name="test", min_sample_size=1)
    framework = ExperimentFramework(config)
    records = [{"decision": "APPROVE", "defaulted": 0, "loan_amount": 10000.0, "confidence": 0.9}]
    result = framework.analyze(records)
    assert result.sample_size["heuristic"] == 0
    assert result.sample_size["ml"] == 0
