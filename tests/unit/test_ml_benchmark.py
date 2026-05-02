from __future__ import annotations

from pathlib import Path

from ml.benchmark.heuristic_vs_ml import benchmark_manifest
from ml.models.calibrate import calibrate_manifest
from ml.models.train import CandidateDefinition, TrainingConfig, run_training


class FakeProbabilityModel:
    def __init__(self, bias: float = 0.0):
        self.bias = bias
        self.feature_importances_: list[float] = []

    def fit(self, X, y):
        rows = X.values.tolist() if hasattr(X, "values") else list(X)
        feature_count = len(rows[0]) if rows else 0
        self.feature_importances_ = [1.0 / feature_count for _ in range(feature_count)] if feature_count else []
        return self

    def predict_proba(self, X):
        rows = X.values.tolist() if hasattr(X, "values") else list(X)
        probabilities = []
        for row in rows:
            signal = row[0] * 0.04 + row[6] * 0.6 + self.bias
            positive_probability = min(max(signal, 0.01), 0.99)
            probabilities.append([1.0 - positive_probability, positive_probability])
        return probabilities


def build_fake_candidate(name: str, bias: float) -> CandidateDefinition:
    return CandidateDefinition(
        name=name,
        family="test-double",
        factory=lambda params: FakeProbabilityModel(**params),
        parameter_grid=[{"bias": bias}],
    )


def test_benchmark_manifest_writes_report(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "accepted.csv"
    _write_training_fixture(dataset_path)
    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(dataset_path))

    config = TrainingConfig(
        experiment_dir=tmp_path / "experiments",
        artifact_dir=tmp_path / "artifacts",
        run_label="benchmark-test",
        include_logistic_regression=False,
        include_xgboost=False,
        include_lightgbm=False,
    )
    training_summary = run_training(
        config,
        candidates=[
            build_fake_candidate("candidate_a", bias=0.0),
            build_fake_candidate("candidate_b", bias=0.1),
        ],
    )
    calibrate_manifest(training_summary.manifest_path, report_dir=tmp_path / "calibration-reports")

    benchmark = benchmark_manifest(
        training_summary.manifest_path,
        report_dir=tmp_path / "benchmark-reports",
    )

    assert benchmark.row_count > 0
    assert benchmark.selected_candidate in {"candidate_a", "candidate_b"}
    assert Path(benchmark.report_path).exists()
    assert benchmark.ab_report["arms"][0]["arm"] == "heuristic"


def _write_training_fixture(path: Path) -> None:
    import csv

    fieldnames = [
        "id",
        "application_type",
        "loan_status",
        "issue_d",
        "annual_inc",
        "loan_amnt",
        "funded_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "verification_status",
        "purpose",
        "dti",
        "delinq_2yrs",
        "earliest_cr_line",
        "fico_range_low",
        "fico_range_high",
        "last_fico_range_low",
        "last_fico_range_high",
        "inq_last_6mths",
        "inq_last_12m",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "collections_12_mths_ex_med",
        "pub_rec_bankruptcies",
        "tax_liens",
        "tot_cur_bal",
        "total_bal_ex_mort",
        "total_rev_hi_lim",
        "total_bc_limit",
        "total_il_high_credit_limit",
        "bc_util",
        "percent_bc_gt_75",
        "all_util",
        "il_util",
        "acc_open_past_24mths",
        "mort_acc",
        "pct_tl_nvr_dlq",
        "total_cu_tl",
        "open_rv_24m",
        "open_il_24m",
        "num_tl_90g_dpd_24m",
        "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op",
    ]
    rows = [
        {
            "id": "1001",
            "application_type": "Individual",
            "loan_status": "Fully Paid",
            "issue_d": "Dec-2015",
            "annual_inc": "120000",
            "loan_amnt": "10000",
            "funded_amnt": "10000",
            "term": " 36 months",
            "int_rate": "8.5",
            "installment": "316",
            "grade": "A",
            "sub_grade": "A3",
            "emp_length": "10+ years",
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "purpose": "credit_card",
            "dti": "8.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2005",
            "fico_range_low": "760",
            "fico_range_high": "764",
            "last_fico_range_low": "780",
            "last_fico_range_high": "784",
            "inq_last_6mths": "0",
            "inq_last_12m": "1",
            "open_acc": "12",
            "pub_rec": "0",
            "revol_bal": "5000",
            "revol_util": "15",
            "total_acc": "24",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "40000",
            "total_bal_ex_mort": "18000",
            "total_rev_hi_lim": "30000",
            "total_bc_limit": "25000",
            "total_il_high_credit_limit": "50000",
            "bc_util": "18",
            "percent_bc_gt_75": "0",
            "all_util": "20",
            "il_util": "25",
            "acc_open_past_24mths": "1",
            "mort_acc": "1",
            "pct_tl_nvr_dlq": "100",
            "total_cu_tl": "1",
            "open_rv_24m": "1",
            "open_il_24m": "0",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "120",
            "mo_sin_rcnt_rev_tl_op": "8",
        },
        {
            "id": "1002",
            "application_type": "Individual",
            "loan_status": "Charged Off",
            "issue_d": "Nov-2015",
            "annual_inc": "45000",
            "loan_amnt": "30000",
            "funded_amnt": "30000",
            "term": " 60 months",
            "int_rate": "19.5",
            "installment": "786",
            "grade": "E",
            "sub_grade": "E2",
            "emp_length": "2 years",
            "home_ownership": "RENT",
            "verification_status": "Source Verified",
            "purpose": "small_business",
            "dti": "28.0",
            "delinq_2yrs": "2",
            "earliest_cr_line": "Jan-2010",
            "fico_range_low": "620",
            "fico_range_high": "624",
            "last_fico_range_low": "560",
            "last_fico_range_high": "564",
            "inq_last_6mths": "4",
            "inq_last_12m": "7",
            "open_acc": "6",
            "pub_rec": "1",
            "revol_bal": "25000",
            "revol_util": "88",
            "total_acc": "12",
            "collections_12_mths_ex_med": "2",
            "pub_rec_bankruptcies": "1",
            "tax_liens": "0",
            "tot_cur_bal": "60000",
            "total_bal_ex_mort": "42000",
            "total_rev_hi_lim": "26000",
            "total_bc_limit": "8000",
            "total_il_high_credit_limit": "25000",
            "bc_util": "91",
            "percent_bc_gt_75": "100",
            "all_util": "82",
            "il_util": "70",
            "acc_open_past_24mths": "6",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "80",
            "total_cu_tl": "0",
            "open_rv_24m": "5",
            "open_il_24m": "3",
            "num_tl_90g_dpd_24m": "2",
            "mo_sin_old_rev_tl_op": "48",
            "mo_sin_rcnt_rev_tl_op": "1",
        },
        {
            "id": "2001",
            "application_type": "Individual",
            "loan_status": "Fully Paid",
            "issue_d": "Jun-2017",
            "annual_inc": "100000",
            "loan_amnt": "12000",
            "funded_amnt": "12000",
            "term": " 36 months",
            "int_rate": "9.9",
            "installment": "386",
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": "5 years",
            "home_ownership": "MORTGAGE",
            "verification_status": "Verified",
            "purpose": "home_improvement",
            "dti": "12.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2008",
            "fico_range_low": "720",
            "fico_range_high": "724",
            "last_fico_range_low": "730",
            "last_fico_range_high": "734",
            "inq_last_6mths": "1",
            "inq_last_12m": "2",
            "open_acc": "11",
            "pub_rec": "0",
            "revol_bal": "8000",
            "revol_util": "22",
            "total_acc": "20",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "45000",
            "total_bal_ex_mort": "16000",
            "total_rev_hi_lim": "28000",
            "total_bc_limit": "22000",
            "total_il_high_credit_limit": "40000",
            "bc_util": "24",
            "percent_bc_gt_75": "0",
            "all_util": "24",
            "il_util": "28",
            "acc_open_past_24mths": "2",
            "mort_acc": "1",
            "pct_tl_nvr_dlq": "98",
            "total_cu_tl": "1",
            "open_rv_24m": "2",
            "open_il_24m": "1",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "110",
            "mo_sin_rcnt_rev_tl_op": "6",
        },
        {
            "id": "3001",
            "application_type": "Individual",
            "loan_status": "Fully Paid",
            "issue_d": "Jul-2018",
            "annual_inc": "90000",
            "loan_amnt": "15000",
            "funded_amnt": "15000",
            "term": " 36 months",
            "int_rate": "11.4",
            "installment": "494",
            "grade": "C",
            "sub_grade": "C1",
            "emp_length": "7 years",
            "home_ownership": "OWN",
            "verification_status": "Verified",
            "purpose": "major_purchase",
            "dti": "14.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2006",
            "fico_range_low": "700",
            "fico_range_high": "704",
            "last_fico_range_low": "710",
            "last_fico_range_high": "714",
            "inq_last_6mths": "1",
            "inq_last_12m": "2",
            "open_acc": "9",
            "pub_rec": "0",
            "revol_bal": "12000",
            "revol_util": "35",
            "total_acc": "18",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "38000",
            "total_bal_ex_mort": "18000",
            "total_rev_hi_lim": "26000",
            "total_bc_limit": "20000",
            "total_il_high_credit_limit": "32000",
            "bc_util": "36",
            "percent_bc_gt_75": "0",
            "all_util": "30",
            "il_util": "32",
            "acc_open_past_24mths": "2",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "97",
            "total_cu_tl": "1",
            "open_rv_24m": "2",
            "open_il_24m": "1",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "100",
            "mo_sin_rcnt_rev_tl_op": "4",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
