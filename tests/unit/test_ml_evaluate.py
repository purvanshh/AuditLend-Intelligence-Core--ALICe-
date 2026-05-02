from __future__ import annotations

import csv
from pathlib import Path

from ml.models.evaluate import (
    compute_confusion_matrix_summary,
    compute_expected_calibration_error,
    evaluate_manifest,
    fairness_analysis,
    summarize_candidate_comparison,
)
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


def write_training_fixture(path: Path) -> None:
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
            "dti": "10.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2007",
            "fico_range_low": "730",
            "fico_range_high": "734",
            "last_fico_range_low": "740",
            "last_fico_range_high": "744",
            "inq_last_6mths": "1",
            "inq_last_12m": "2",
            "open_acc": "11",
            "pub_rec": "0",
            "revol_bal": "6000",
            "revol_util": "22",
            "total_acc": "20",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "45000",
            "total_bal_ex_mort": "20000",
            "total_rev_hi_lim": "32000",
            "total_bc_limit": "22000",
            "total_il_high_credit_limit": "42000",
            "bc_util": "20",
            "percent_bc_gt_75": "0",
            "all_util": "24",
            "il_util": "27",
            "acc_open_past_24mths": "2",
            "mort_acc": "1",
            "pct_tl_nvr_dlq": "99",
            "total_cu_tl": "1",
            "open_rv_24m": "2",
            "open_il_24m": "1",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "110",
            "mo_sin_rcnt_rev_tl_op": "6",
        },
        {
            "id": "2002",
            "application_type": "Individual",
            "loan_status": "Charged Off",
            "issue_d": "Jul-2017",
            "annual_inc": "50000",
            "loan_amnt": "28000",
            "funded_amnt": "28000",
            "term": " 60 months",
            "int_rate": "18.2",
            "installment": "714",
            "grade": "D",
            "sub_grade": "D4",
            "emp_length": "1 year",
            "home_ownership": "RENT",
            "verification_status": "Not Verified",
            "purpose": "debt_consolidation",
            "dti": "26.0",
            "delinq_2yrs": "1",
            "earliest_cr_line": "Jan-2012",
            "fico_range_low": "640",
            "fico_range_high": "644",
            "last_fico_range_low": "570",
            "last_fico_range_high": "574",
            "inq_last_6mths": "3",
            "inq_last_12m": "6",
            "open_acc": "8",
            "pub_rec": "0",
            "revol_bal": "23000",
            "revol_util": "79",
            "total_acc": "15",
            "collections_12_mths_ex_med": "1",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "1",
            "tot_cur_bal": "52000",
            "total_bal_ex_mort": "36000",
            "total_rev_hi_lim": "24000",
            "total_bc_limit": "9000",
            "total_il_high_credit_limit": "30000",
            "bc_util": "84",
            "percent_bc_gt_75": "85",
            "all_util": "74",
            "il_util": "66",
            "acc_open_past_24mths": "5",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "87",
            "total_cu_tl": "0",
            "open_rv_24m": "4",
            "open_il_24m": "2",
            "num_tl_90g_dpd_24m": "1",
            "mo_sin_old_rev_tl_op": "56",
            "mo_sin_rcnt_rev_tl_op": "2",
        },
        {
            "id": "3001",
            "application_type": "Individual",
            "loan_status": "Fully Paid",
            "issue_d": "Feb-2018",
            "annual_inc": "98000",
            "loan_amnt": "14000",
            "funded_amnt": "14000",
            "term": " 36 months",
            "int_rate": "10.5",
            "installment": "455",
            "grade": "B",
            "sub_grade": "B4",
            "emp_length": "6 years",
            "home_ownership": "OWN",
            "verification_status": "Verified",
            "purpose": "major_purchase",
            "dti": "11.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2006",
            "fico_range_low": "725",
            "fico_range_high": "729",
            "last_fico_range_low": "738",
            "last_fico_range_high": "742",
            "inq_last_6mths": "1",
            "inq_last_12m": "2",
            "open_acc": "10",
            "pub_rec": "0",
            "revol_bal": "7000",
            "revol_util": "24",
            "total_acc": "18",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "48000",
            "total_bal_ex_mort": "21000",
            "total_rev_hi_lim": "31000",
            "total_bc_limit": "23000",
            "total_il_high_credit_limit": "41000",
            "bc_util": "23",
            "percent_bc_gt_75": "0",
            "all_util": "26",
            "il_util": "29",
            "acc_open_past_24mths": "2",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "99",
            "total_cu_tl": "1",
            "open_rv_24m": "2",
            "open_il_24m": "1",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "112",
            "mo_sin_rcnt_rev_tl_op": "5",
        },
        {
            "id": "3002",
            "application_type": "Individual",
            "loan_status": "Charged Off",
            "issue_d": "Mar-2018",
            "annual_inc": "52000",
            "loan_amnt": "26000",
            "funded_amnt": "26000",
            "term": " 60 months",
            "int_rate": "17.8",
            "installment": "658",
            "grade": "D",
            "sub_grade": "D2",
            "emp_length": "3 years",
            "home_ownership": "RENT",
            "verification_status": "Source Verified",
            "purpose": "credit_card",
            "dti": "25.0",
            "delinq_2yrs": "1",
            "earliest_cr_line": "Jan-2011",
            "fico_range_low": "650",
            "fico_range_high": "654",
            "last_fico_range_low": "580",
            "last_fico_range_high": "584",
            "inq_last_6mths": "3",
            "inq_last_12m": "5",
            "open_acc": "7",
            "pub_rec": "0",
            "revol_bal": "21000",
            "revol_util": "76",
            "total_acc": "14",
            "collections_12_mths_ex_med": "1",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "50000",
            "total_bal_ex_mort": "34000",
            "total_rev_hi_lim": "23000",
            "total_bc_limit": "8500",
            "total_il_high_credit_limit": "29000",
            "bc_util": "81",
            "percent_bc_gt_75": "78",
            "all_util": "71",
            "il_util": "64",
            "acc_open_past_24mths": "4",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "88",
            "total_cu_tl": "0",
            "open_rv_24m": "4",
            "open_il_24m": "2",
            "num_tl_90g_dpd_24m": "1",
            "mo_sin_old_rev_tl_op": "58",
            "mo_sin_rcnt_rev_tl_op": "2",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_expected_calibration_error_is_zero_for_perfect_bins():
    summary = compute_expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], bins=2)

    assert summary.ece == 0.0
    assert summary.max_calibration_gap == 0.0


def test_compute_confusion_matrix_summary_matches_expected_counts():
    summary = compute_confusion_matrix_summary([0, 1, 1, 0], [0.2, 0.8, 0.55, 0.9], threshold=0.6)

    assert summary.true_positive == 1
    assert summary.false_positive == 1
    assert summary.true_negative == 1
    assert summary.false_negative == 1


def test_summarize_candidate_comparison_keeps_best_row_per_candidate():
    rows = [
        {
            "candidate_name": "xgboost",
            "family": "xgboost",
            "params": {"max_depth": 4},
            "validation_metrics": {"auc_pr": 0.7, "auc_roc": 0.8, "brier_score": 0.2},
            "test_metrics": {"auc_pr": 0.68, "auc_roc": 0.79, "brier_score": 0.21},
        },
        {
            "candidate_name": "xgboost",
            "family": "xgboost",
            "params": {"max_depth": 6},
            "validation_metrics": {"auc_pr": 0.8, "auc_roc": 0.81, "brier_score": 0.19},
            "test_metrics": {"auc_pr": 0.72, "auc_roc": 0.8, "brier_score": 0.2},
        },
        {
            "candidate_name": "lightgbm",
            "family": "lightgbm",
            "params": {"num_leaves": 31},
            "validation_metrics": {"auc_pr": 0.78, "auc_roc": 0.82, "brier_score": 0.18},
            "test_metrics": {"auc_pr": 0.7, "auc_roc": 0.79, "brier_score": 0.19},
        },
    ]

    summaries = summarize_candidate_comparison(rows)

    assert summaries[0].candidate_name == "xgboost"
    assert summaries[0].params == {"max_depth": 6}
    assert len(summaries) == 2


def test_evaluate_manifest_writes_report(monkeypatch, tmp_path):
    dataset_path = tmp_path / "accepted.csv"
    write_training_fixture(dataset_path)
    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(dataset_path))

    config = TrainingConfig(
        experiment_dir=tmp_path / "experiments",
        artifact_dir=tmp_path / "artifacts",
        run_label="eval-test",
        include_logistic_regression=False,
        include_xgboost=False,
        include_lightgbm=False,
    )
    summary = run_training(
        config,
        candidates=[
            build_fake_candidate("candidate_a", bias=0.0),
            build_fake_candidate("candidate_b", bias=0.1),
        ],
    )

    report = evaluate_manifest(summary.manifest_path, report_dir=tmp_path / "reports")

    assert Path(report.report_path).exists()
    assert report.selected_candidate in {"candidate_a", "candidate_b"}
    assert report.evaluation_splits["test"]["metrics"]["row_count"] == 2
    assert len(report.candidate_comparison) == 2


def test_fairness_analysis_computes_reference_disparities():
    rows = [
        {"zip_code_prefix": "111", "target_defaulted": 0},
        {"zip_code_prefix": "111", "target_defaulted": 0},
        {"zip_code_prefix": "111", "target_defaulted": 1},
        {"zip_code_prefix": "222", "target_defaulted": 0},
        {"zip_code_prefix": "222", "target_defaulted": 1},
        {"zip_code_prefix": "222", "target_defaulted": 1},
    ]
    probabilities = [0.10, 0.20, 0.70, 0.15, 0.55, 0.80]

    summary = fairness_analysis(
        rows,
        probabilities,
        "zip_code_prefix",
        threshold=0.5,
        min_count=1,
    )

    assert summary.reference_group == "111"
    assert summary.reference_approval_rate == 0.666667
    assert summary.reference_true_positive_rate == 1.0
    assert summary.max_abs_statistical_parity_difference == 0.333333
    assert summary.max_abs_equal_opportunity_difference == 0.0
    assert [group.group_value for group in summary.groups] == ["111", "222"]
