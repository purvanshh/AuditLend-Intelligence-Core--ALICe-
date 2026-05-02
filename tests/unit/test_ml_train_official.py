from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.models import train as train_module


def _raw_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "1",
                "loan_amnt": 10000,
                "funded_amnt": 10000,
                "term": " 36 months",
                "int_rate": 8.5,
                "installment": 316,
                "grade": "A",
                "sub_grade": "A1",
                "emp_length": "10+ years",
                "home_ownership": "MORTGAGE",
                "annual_inc": 120000,
                "verification_status": "Verified",
                "issue_d": "Dec-2016",
                "loan_status": "Fully Paid",
                "purpose": "credit_card",
                "application_type": "Individual",
                "dti": 10.0,
                "delinq_2yrs": 0,
                "earliest_cr_line": "Jan-2000",
                "fico_range_low": 760,
                "fico_range_high": 764,
                "last_fico_range_low": 770,
                "last_fico_range_high": 774,
                "inq_last_6mths": 1,
                "inq_last_12m": 2,
                "open_acc": 10,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": 15,
                "total_acc": 20,
                "collections_12_mths_ex_med": 0,
                "pub_rec_bankruptcies": 0,
                "tax_liens": 0,
                "tot_cur_bal": 40000,
                "total_bal_ex_mort": 15000,
                "total_rev_hi_lim": 30000,
                "total_bc_limit": 20000,
                "total_il_high_credit_limit": 50000,
                "bc_util": 20,
                "percent_bc_gt_75": 0,
                "all_util": 25,
                "il_util": 18,
                "acc_open_past_24mths": 1,
                "mort_acc": 1,
                "pct_tl_nvr_dlq": 98,
                "total_cu_tl": 0,
                "open_rv_24m": 1,
                "open_il_24m": 1,
                "num_tl_90g_dpd_24m": 0,
                "mo_sin_old_rev_tl_op": 120,
                "mo_sin_rcnt_rev_tl_op": 8,
            },
            {
                "id": "2",
                "loan_amnt": 5000,
                "funded_amnt": 5000,
                "term": " 36 months",
                "int_rate": 15.0,
                "installment": 173,
                "grade": "C",
                "sub_grade": "C2",
                "emp_length": "< 1 year",
                "home_ownership": "RENT",
                "annual_inc": 50000,
                "verification_status": "Source Verified",
                "issue_d": "Jan-2018",
                "loan_status": "Charged Off",
                "purpose": "small_business",
                "application_type": "Joint App",
                "dti": 22.0,
                "delinq_2yrs": 1,
                "earliest_cr_line": "Jan-2012",
                "fico_range_low": 640,
                "fico_range_high": 644,
                "last_fico_range_low": 610,
                "last_fico_range_high": 614,
                "inq_last_6mths": 3,
                "inq_last_12m": 6,
                "open_acc": 6,
                "pub_rec": 1,
                "revol_bal": 10000,
                "revol_util": 70,
                "total_acc": 12,
                "collections_12_mths_ex_med": 1,
                "pub_rec_bankruptcies": 1,
                "tax_liens": 1,
                "tot_cur_bal": 30000,
                "total_bal_ex_mort": 18000,
                "total_rev_hi_lim": 10000,
                "total_bc_limit": 6000,
                "total_il_high_credit_limit": 15000,
                "bc_util": 80,
                "percent_bc_gt_75": 100,
                "all_util": 65,
                "il_util": 45,
                "acc_open_past_24mths": 5,
                "mort_acc": 0,
                "pct_tl_nvr_dlq": 80,
                "total_cu_tl": 1,
                "open_rv_24m": 2,
                "open_il_24m": 2,
                "num_tl_90g_dpd_24m": 1,
                "mo_sin_old_rev_tl_op": 48,
                "mo_sin_rcnt_rev_tl_op": 2,
            },
        ]
    )


def _feature_frame(issue_year: int, target: int, loan_id: str) -> pd.DataFrame:
    row = {column: 0.0 for column in train_module.OFFICIAL_INPUT_FEATURES}
    row.update(
        {
            "grade": "A" if target == 0 else "D",
            "sub_grade": "A1" if target == 0 else "D4",
            "purpose": "credit_card",
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "loan_amount": 10000.0 + (target * 5000.0),
            "loan_amount_to_income": 0.2 + (target * 0.4),
            "installment_to_income": 0.05 + (target * 0.15),
            "credit_score_midpoint": 780.0 - (target * 120.0),
            "credit_score_recent_delta": 5.0 - (target * 20.0),
            "revol_util_ratio": 0.1 + (target * 0.5),
            "bc_util_ratio": 0.1 + (target * 0.5),
            "all_util_ratio": 0.15 + (target * 0.45),
            "il_util_ratio": 0.05 + (target * 0.35),
            "existing_emi_to_income": 0.05 + (target * 0.2),
            "monthly_income": 8000.0,
            "term_months": 36.0,
        }
    )
    row["loan_id"] = loan_id
    row["issue_date"] = datetime(issue_year, 1, 1)
    row["target_defaulted"] = target
    return pd.DataFrame([row])


class FakeOfficialModel:
    def __init__(self, params: dict, seed: int = 42) -> None:
        self.params = dict(params)
        self.seed = seed
        self.feature_importances_ = [0.1] * len(train_module.OFFICIAL_INPUT_FEATURES)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        values = []
        for _, row in X.iterrows():
            risk = 0.2 if float(row["loan_amount_to_income"]) < 0.5 else 0.8
            values.append([1.0 - risk, risk])
        return values

    def get_feature_names(self) -> list[str]:
        return list(train_module.OFFICIAL_INPUT_FEATURES)


def test_build_official_feature_chunk_filters_and_engineers_rows() -> None:
    feature_chunk = train_module._build_official_feature_chunk(_raw_training_frame())

    assert list(feature_chunk["loan_id"]) == ["1"]
    assert feature_chunk.iloc[0]["grade"] == "A"
    assert round(float(feature_chunk.iloc[0]["loan_amount_to_income"]), 4) == 1.0
    assert feature_chunk.iloc[0]["target_defaulted"] == 0


def test_prepare_official_training_dataset_groups_rows_by_split(monkeypatch, tmp_path: Path) -> None:
    raw_frame = pd.concat(
        [
            _raw_training_frame().assign(issue_d="Dec-2016", loan_status="Fully Paid", application_type="Individual", id="1"),
            _raw_training_frame().iloc[[0]].assign(issue_d="Jan-2017", id="3"),
            _raw_training_frame().iloc[[0]].assign(issue_d="Jan-2018", id="4"),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(train_module, "ensure_lending_club_data_path", lambda env_var="LENDING_CLUB_DATA_PATH": tmp_path / "data.csv")
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: [raw_frame])

    prepared = train_module.prepare_official_training_dataset()

    assert prepared.split_counts["train"] == 2
    assert prepared.split_counts["validation"] == 1
    assert prepared.split_counts["test"] == 1
    assert prepared.split_counts["holdout"] == 0


def test_train_official_xgb_v1_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    train_frame = pd.concat(
        [
            _feature_frame(2016, 0, "t1"),
            _feature_frame(2016, 1, "t2"),
            _feature_frame(2016, 0, "t3"),
            _feature_frame(2016, 1, "t4"),
        ],
        ignore_index=True,
    )
    validation_frame = pd.concat(
        [_feature_frame(2017, 0, "v1"), _feature_frame(2017, 1, "v2"), _feature_frame(2017, 0, "v3"), _feature_frame(2017, 1, "v4")],
        ignore_index=True,
    )
    test_frame = pd.concat(
        [_feature_frame(2018, 0, "x1"), _feature_frame(2018, 1, "x2"), _feature_frame(2018, 0, "x3"), _feature_frame(2018, 1, "x4")],
        ignore_index=True,
    )
    prepared = train_module.OfficialPreparedDataset(
        split_frames={"train": train_frame, "validation": validation_frame, "test": test_frame, "holdout": test_frame.iloc[0:0].copy()},
        feature_columns=list(train_module.OFFICIAL_INPUT_FEATURES),
        split_counts={"train": 4, "validation": 4, "test": 4, "holdout": 0},
        data_hash="abc123",
    )

    monkeypatch.setattr(train_module, "prepare_official_training_dataset", lambda env_var="LENDING_CLUB_DATA_PATH": prepared)
    monkeypatch.setattr(train_module, "ensure_lending_club_data_path", lambda env_var="LENDING_CLUB_DATA_PATH": tmp_path / "full.csv")
    monkeypatch.setattr(train_module, "OfficialXGBV1Model", FakeOfficialModel)
    monkeypatch.setattr(train_module, "OFFICIAL_MODEL_ARTIFACT_PATH", tmp_path / "XGB_V1_model.pkl")
    monkeypatch.setattr(train_module, "OFFICIAL_CALIBRATOR_ARTIFACT_PATH", tmp_path / "XGB_V1_calibrator.pkl")
    monkeypatch.setattr(train_module, "OFFICIAL_FEATURE_SPEC_PATH", tmp_path / "XGB_V1_features.json")
    monkeypatch.setattr(train_module, "OFFICIAL_MANIFEST_PATH", tmp_path / "manifest.yaml")
    monkeypatch.setattr(train_module, "OFFICIAL_SEARCH_RESULTS_PATH", tmp_path / "XGB_V1_search_results.jsonl")

    summary = train_module.train_official_xgb_v1()

    assert summary.model_version == "XGB_V1"
    assert (tmp_path / "XGB_V1_model.pkl").exists()
    assert (tmp_path / "XGB_V1_calibrator.pkl").exists()
    assert (tmp_path / "XGB_V1_features.json").exists()
    assert (tmp_path / "manifest.yaml").exists()
    assert summary.split_counts["test"] == 4
