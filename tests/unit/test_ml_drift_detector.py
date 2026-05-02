from __future__ import annotations

from prometheus_client import generate_latest

from ml.governance.drift_detector import (
    build_reference_feature_snapshot,
    detect_feature_drift,
    detect_feature_drift_from_snapshot,
)


def test_detect_feature_drift_flags_shifted_feature() -> None:
    reference_rows = [
        {"loan_amount": 10_000.0, "dti_ratio": 0.10},
        {"loan_amount": 12_000.0, "dti_ratio": 0.11},
        {"loan_amount": 11_500.0, "dti_ratio": 0.12},
        {"loan_amount": 13_000.0, "dti_ratio": 0.10},
        {"loan_amount": 12_500.0, "dti_ratio": 0.09},
    ] * 20
    candidate_rows = [
        {"loan_amount": 45_000.0, "dti_ratio": 0.10},
        {"loan_amount": 48_000.0, "dti_ratio": 0.11},
        {"loan_amount": 47_500.0, "dti_ratio": 0.12},
        {"loan_amount": 49_000.0, "dti_ratio": 0.10},
        {"loan_amount": 46_500.0, "dti_ratio": 0.09},
    ] * 20

    report = detect_feature_drift(
        reference_rows,
        candidate_rows,
        feature_names=("loan_amount", "dti_ratio"),
        model_version="XGB_V1",
    )

    assert report.alert_count == 1
    assert report.drifted_features[0].feature_name == "loan_amount"

    metrics = generate_latest().decode("utf-8")
    assert 'auditlend_drift_alerts_total{feature="loan_amount",model_version="XGB_V1"}' in metrics


def test_detect_feature_drift_from_snapshot_reuses_saved_distribution() -> None:
    reference_rows = [
        {"credit_score_midpoint": 720.0},
        {"credit_score_midpoint": 725.0},
        {"credit_score_midpoint": 730.0},
        {"credit_score_midpoint": 735.0},
    ] * 25
    candidate_rows = [
        {"credit_score_midpoint": 620.0},
        {"credit_score_midpoint": 625.0},
        {"credit_score_midpoint": 630.0},
        {"credit_score_midpoint": 635.0},
    ] * 25

    snapshot = build_reference_feature_snapshot(reference_rows, feature_names=("credit_score_midpoint",))
    report = detect_feature_drift_from_snapshot(
        snapshot,
        candidate_rows,
        model_version="XGB_V2",
    )

    assert report.total_features == 1
    assert report.alert_count == 1
    assert report.to_audit_payload()["drifted_features"][0]["feature_name"] == "credit_score_midpoint"
