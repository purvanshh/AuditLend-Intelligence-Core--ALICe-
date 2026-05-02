from ml.governance.ab_test import OutcomeRecord, assign_experiment_arm, summarize_outcomes


def test_assign_experiment_arm_is_deterministic() -> None:
    first = assign_experiment_arm("application-123", ml_ratio=0.2, enabled=True)
    second = assign_experiment_arm("application-123", ml_ratio=0.2, enabled=True)

    assert first == second
    assert first.enabled is True
    assert first.arm in {"heuristic", "ml"}
    assert 0 <= first.bucket < 10_000


def test_summarize_outcomes_computes_arm_deltas() -> None:
    report = summarize_outcomes(
        [
            OutcomeRecord("heuristic", "APPROVE", 0.90, 0, 10_000.0, "heuristic"),
            OutcomeRecord("heuristic", "APPROVE", 0.85, 1, 10_000.0, "heuristic"),
            OutcomeRecord("ml", "APPROVE", 0.92, 0, 10_000.0, "ml"),
            OutcomeRecord("ml", "DECLINE", 0.88, 1, 10_000.0, "ml"),
        ],
        ml_ratio=0.5,
    )

    assert report.approval_rate_delta_ml_minus_heuristic == -0.5
    assert report.default_rate_delta_ml_minus_heuristic == -0.5
    assert report.profit_delta_ml_minus_heuristic == 6500.0
