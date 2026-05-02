from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ml import benchmark as benchmark_package
from ml.benchmark import heuristic_vs_ml as benchmark_module


class FakeCalibrator:
    def predict(self, values):
        return values


class FakeModel:
    def predict_proba(self, X):
        probabilities = []
        for _, row in X.iterrows():
            probability = 0.2 if float(row["loan_amount_to_income"]) < 0.5 else 0.8
            probabilities.append([1.0 - probability, probability])
        return probabilities


def _official_frame() -> pd.DataFrame:
    base = {column: 0.0 for column in benchmark_module.OFFICIAL_INPUT_FEATURES}
    rows = []
    for loan_id, ratio, target in [("1", 0.2, 0), ("2", 0.7, 1), ("3", 0.25, 0), ("4", 0.8, 1)]:
        row = dict(base)
        row.update(
            {
                "grade": "A",
                "sub_grade": "A1",
                "purpose": "credit_card",
                "home_ownership": "MORTGAGE",
                "verification_status": "Verified",
                "loan_amount": 10000.0,
                "monthly_income": 8000.0,
                "loan_amount_to_income": ratio,
                "installment_to_income": ratio / 4,
                "existing_emi_to_income": ratio / 5,
                "credit_score_midpoint": 780.0 if target == 0 else 620.0,
                "all_util_ratio": ratio,
                "credit_card_headroom_ratio": 1.0 - ratio,
                "never_delinquent_ratio": 0.95 if target == 0 else 0.7,
                "recent_inquiry_pressure": 0.05 if target == 0 else 0.2,
            }
        )
        row["loan_id"] = loan_id
        row["issue_date"] = datetime(2018, 1, 1)
        row["target_defaulted"] = target
        rows.append(row)
    return pd.DataFrame(rows)


def test_benchmark_official_xgb_v1_writes_report(monkeypatch, tmp_path: Path) -> None:
    frame = _official_frame()
    prepared = benchmark_module.prepare_official_training_dataset
    monkeypatch.setattr(
        benchmark_module,
        "prepare_official_training_dataset",
        lambda env_var="LENDING_CLUB_DATA_PATH": type(
            "Prepared",
            (),
            {"split_frames": {"test": frame}},
        )(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_manifest",
        lambda path: {"model_version": "XGB_V1", "model_artifact_path": str(tmp_path / "model.pkl"), "calibrator_artifact_path": str(tmp_path / "cal.pkl")},
    )
    monkeypatch.setattr(benchmark_module, "load_model_artifact", lambda path: FakeModel() if str(path).endswith("model.pkl") else FakeCalibrator())
    monkeypatch.setattr(benchmark_module, "predict_probabilities", lambda model, X: [row[1] for row in model.predict_proba(X)])

    report = benchmark_module.benchmark_official_xgb_v1(report_dir=tmp_path)

    assert report.run_id == "XGB_V1"
    assert Path(report.report_path).exists()
    assert report.ab_report["arms"][0]["row_count"] == 4

    benchmark_class = benchmark_package.BenchmarkReport
    benchmark_fn = benchmark_package.benchmark_manifest
    assert benchmark_class.__name__ == "BenchmarkReport"
    assert callable(benchmark_fn)

    monkeypatch.setattr(benchmark_module, "prepare_official_training_dataset", prepared)
