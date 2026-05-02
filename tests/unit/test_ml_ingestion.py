from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml.data.ingestion import (
    DEFAULT_LENDING_CLUB_DATA_PATH,
    clean_lending_club_row,
    ensure_lending_club_data_path,
    load_lending_club_data,
    profile_lending_club_data,
    resolve_lending_club_data_path,
)


def test_resolve_lending_club_data_path_uses_default_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("LENDING_CLUB_DATA_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    expected = (tmp_path / DEFAULT_LENDING_CLUB_DATA_PATH).resolve()

    assert resolve_lending_club_data_path() == expected


def test_ensure_lending_club_data_path_respects_env_override(monkeypatch, tmp_path):
    dataset_path = tmp_path / "custom-dataset.csv"
    dataset_path.write_text("loan_amnt\n1000\n", encoding="utf-8")
    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(dataset_path))

    assert ensure_lending_club_data_path() == dataset_path.resolve()


def test_ensure_lending_club_data_path_raises_clear_error_when_missing(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.csv.gz"
    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(missing_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        ensure_lending_club_data_path()

    assert "LENDING_CLUB_DATA_PATH" in str(exc_info.value)
    assert str(DEFAULT_LENDING_CLUB_DATA_PATH) in str(exc_info.value)


def test_clean_lending_club_row_filters_non_modelable_records():
    joint_row = {
        "id": "1",
        "application_type": "Joint App",
        "loan_status": "Fully Paid",
        "issue_d": "Dec-2015",
        "annual_inc": "60000",
        "loan_amnt": "10000",
        "term": " 36 months",
    }
    current_row = {
        "id": "2",
        "application_type": "Individual",
        "loan_status": "Current",
        "issue_d": "Dec-2015",
        "annual_inc": "60000",
        "loan_amnt": "10000",
        "term": " 36 months",
    }

    assert clean_lending_club_row(joint_row) is None
    assert clean_lending_club_row(current_row) is None


def test_clean_lending_club_row_maps_core_fields():
    raw_row = {
        "id": "3",
        "application_type": "Individual",
        "loan_status": "Charged Off",
        "issue_d": "Jan-2016",
        "annual_inc": "120000",
        "loan_amnt": "24000",
        "funded_amnt": "24000",
        "term": " 60 months",
        "int_rate": "10.50",
        "installment": "515.81",
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": "18.0",
        "delinq_2yrs": "1",
        "earliest_cr_line": "Jan-2006",
        "fico_range_low": "690",
        "fico_range_high": "694",
        "last_fico_range_low": "620",
        "last_fico_range_high": "624",
        "inq_last_6mths": "2",
        "inq_last_12m": "3",
        "open_acc": "12",
        "pub_rec": "0",
        "revol_bal": "25000",
        "revol_util": "52.0",
        "total_acc": "24",
        "collections_12_mths_ex_med": "0",
        "pub_rec_bankruptcies": "0",
        "tax_liens": "0",
        "tot_cur_bal": "140000",
        "total_bal_ex_mort": "50000",
        "total_rev_hi_lim": "48000",
        "total_bc_limit": "32000",
        "total_il_high_credit_limit": "70000",
        "bc_util": "48.0",
        "percent_bc_gt_75": "12.5",
        "all_util": "46.0",
        "il_util": "38.0",
        "acc_open_past_24mths": "4",
        "mort_acc": "1",
        "pct_tl_nvr_dlq": "95.0",
        "total_cu_tl": "2",
        "open_rv_24m": "3",
        "open_il_24m": "2",
        "num_tl_90g_dpd_24m": "0",
        "mo_sin_old_rev_tl_op": "96",
        "mo_sin_rcnt_rev_tl_op": "5",
    }

    cleaned = clean_lending_club_row(raw_row)

    assert cleaned is not None
    assert cleaned["defaulted"] == 1
    assert cleaned["term_months"] == 60
    assert cleaned["monthly_income"] == 10000.0
    assert cleaned["estimated_existing_emi"] == 1800.0
    assert cleaned["fico_midpoint"] == 692.0
    assert cleaned["last_fico_midpoint"] == 622.0


def test_load_lending_club_data_and_profile_from_small_fixture(monkeypatch, tmp_path):
    dataset_path = tmp_path / "accepted.csv"
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
            "id": "10",
            "application_type": "Individual",
            "loan_status": "Fully Paid",
            "issue_d": "Dec-2015",
            "annual_inc": "60000",
            "loan_amnt": "10000",
            "funded_amnt": "10000",
            "term": " 36 months",
            "int_rate": "9.99",
            "installment": "322.0",
            "grade": "B",
            "sub_grade": "B2",
            "emp_length": "2 years",
            "home_ownership": "RENT",
            "verification_status": "Verified",
            "purpose": "credit_card",
            "dti": "12.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2010",
            "fico_range_low": "680",
            "fico_range_high": "684",
            "last_fico_range_low": "700",
            "last_fico_range_high": "704",
            "inq_last_6mths": "1",
            "inq_last_12m": "2",
            "open_acc": "8",
            "pub_rec": "0",
            "revol_bal": "5000",
            "revol_util": "25.0",
            "total_acc": "14",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "30000",
            "total_bal_ex_mort": "15000",
            "total_rev_hi_lim": "20000",
            "total_bc_limit": "12000",
            "total_il_high_credit_limit": "25000",
            "bc_util": "20.0",
            "percent_bc_gt_75": "0.0",
            "all_util": "28.0",
            "il_util": "30.0",
            "acc_open_past_24mths": "2",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "100.0",
            "total_cu_tl": "1",
            "open_rv_24m": "2",
            "open_il_24m": "1",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "72",
            "mo_sin_rcnt_rev_tl_op": "4",
        },
        {
            "id": "11",
            "application_type": "Individual",
            "loan_status": "Charged Off",
            "issue_d": "Jan-2018",
            "annual_inc": "90000",
            "loan_amnt": "25000",
            "funded_amnt": "25000",
            "term": " 60 months",
            "int_rate": "14.99",
            "installment": "590.0",
            "grade": "C",
            "sub_grade": "C4",
            "emp_length": "10+ years",
            "home_ownership": "MORTGAGE",
            "verification_status": "Source Verified",
            "purpose": "debt_consolidation",
            "dti": "20.0",
            "delinq_2yrs": "1",
            "earliest_cr_line": "Jan-2005",
            "fico_range_low": "660",
            "fico_range_high": "664",
            "last_fico_range_low": "560",
            "last_fico_range_high": "564",
            "inq_last_6mths": "2",
            "inq_last_12m": "4",
            "open_acc": "12",
            "pub_rec": "0",
            "revol_bal": "18000",
            "revol_util": "68.0",
            "total_acc": "24",
            "collections_12_mths_ex_med": "1",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "90000",
            "total_bal_ex_mort": "44000",
            "total_rev_hi_lim": "30000",
            "total_bc_limit": "15000",
            "total_il_high_credit_limit": "50000",
            "bc_util": "72.0",
            "percent_bc_gt_75": "50.0",
            "all_util": "61.0",
            "il_util": "45.0",
            "acc_open_past_24mths": "5",
            "mort_acc": "1",
            "pct_tl_nvr_dlq": "92.0",
            "total_cu_tl": "2",
            "open_rv_24m": "3",
            "open_il_24m": "2",
            "num_tl_90g_dpd_24m": "1",
            "mo_sin_old_rev_tl_op": "110",
            "mo_sin_rcnt_rev_tl_op": "3",
        },
        {
            "id": "12",
            "application_type": "Joint App",
            "loan_status": "Fully Paid",
            "issue_d": "Jan-2018",
            "annual_inc": "100000",
            "loan_amnt": "15000",
            "funded_amnt": "15000",
            "term": " 36 months",
            "int_rate": "12.99",
            "installment": "500.0",
            "grade": "C",
            "sub_grade": "C1",
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "verification_status": "Verified",
            "purpose": "home_improvement",
            "dti": "18.0",
            "delinq_2yrs": "0",
            "earliest_cr_line": "Jan-2008",
            "fico_range_low": "700",
            "fico_range_high": "704",
            "last_fico_range_low": "710",
            "last_fico_range_high": "714",
            "inq_last_6mths": "0",
            "inq_last_12m": "1",
            "open_acc": "10",
            "pub_rec": "0",
            "revol_bal": "7000",
            "revol_util": "30.0",
            "total_acc": "16",
            "collections_12_mths_ex_med": "0",
            "pub_rec_bankruptcies": "0",
            "tax_liens": "0",
            "tot_cur_bal": "40000",
            "total_bal_ex_mort": "17000",
            "total_rev_hi_lim": "22000",
            "total_bc_limit": "11000",
            "total_il_high_credit_limit": "26000",
            "bc_util": "31.0",
            "percent_bc_gt_75": "0.0",
            "all_util": "29.0",
            "il_util": "33.0",
            "acc_open_past_24mths": "3",
            "mort_acc": "0",
            "pct_tl_nvr_dlq": "98.0",
            "total_cu_tl": "1",
            "open_rv_24m": "1",
            "open_il_24m": "1",
            "num_tl_90g_dpd_24m": "0",
            "mo_sin_old_rev_tl_op": "85",
            "mo_sin_rcnt_rev_tl_op": "7",
        },
    ]
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setenv("LENDING_CLUB_DATA_PATH", str(dataset_path))

    cleaned_rows = load_lending_club_data()
    profile = profile_lending_club_data()

    assert len(cleaned_rows) == 2
    assert profile.total_rows == 3
    assert profile.individual_rows == 2
    assert profile.modeled_rows == 2
    assert profile.defaulted_rows == 1
    assert profile.issue_date_min == "2015-12-01"
    assert profile.issue_date_max == "2018-01-01"
    assert profile.split_counts == {"train": 1, "test": 1}
