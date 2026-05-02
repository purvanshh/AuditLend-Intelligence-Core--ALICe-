from __future__ import annotations

from datetime import date

from ml.data.features import build_feature_row, compute_correlation_matrix


def test_build_feature_row_generates_expected_ratios():
    clean_row = {
        "loan_id": "100",
        "issue_date": date(2018, 1, 1),
        "loan_status": "Fully Paid",
        "grade": "B",
        "sub_grade": "B1",
        "purpose": "credit_card",
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "loan_amount": 12000.0,
        "funded_amount": 12000.0,
        "term_months": 36,
        "interest_rate_pct": 9.5,
        "installment": 385.0,
        "monthly_income": 6000.0,
        "estimated_existing_emi": 900.0,
        "dti_pct": 15.0,
        "fico_midpoint": 700.0,
        "last_fico_midpoint": 720.0,
        "employment_length_years": 5.0,
        "earliest_credit_line": date(2008, 1, 1),
        "revol_util_pct": 40.0,
        "bc_util_pct": 50.0,
        "all_util_pct": 35.0,
        "il_util_pct": 20.0,
        "revol_bal": 10000.0,
        "tot_cur_bal": 50000.0,
        "total_bal_ex_mort": 22000.0,
        "total_rev_hi_lim": 25000.0,
        "total_bc_limit": 20000.0,
        "delinq_2yrs": 1.0,
        "inq_last_6mths": 3.0,
        "inq_last_12m": 5.0,
        "open_acc": 10.0,
        "total_acc": 20.0,
        "mort_acc": 1.0,
        "pub_rec_bankruptcies": 0.0,
        "tax_liens": 1.0,
        "percent_bc_gt_75": 10.0,
        "pct_tl_nvr_dlq": 96.0,
        "collections_12_mths_ex_med": 2.0,
        "mo_sin_rcnt_rev_tl_op": 4.0,
        "mo_sin_old_rev_tl_op": 120.0,
        "open_rv_24m": 2.0,
        "open_il_24m": 1.0,
        "defaulted": 0,
    }

    feature_row = build_feature_row(clean_row)

    assert feature_row["loan_amount_to_income"] == 2.0
    assert round(feature_row["installment_to_income"], 4) == round(385.0 / 6000.0, 4)
    assert feature_row["credit_score_recent_delta"] == 20.0
    assert feature_row["credit_history_age_years"] == 10.0
    assert feature_row["tax_lien_flag"] == 1.0
    assert feature_row["target_defaulted"] == 0.0


def test_compute_correlation_matrix_returns_identity_on_diagonal():
    feature_rows = [
        {"loan_amount": 1.0, "monthly_income": 2.0},
        {"loan_amount": 2.0, "monthly_income": 4.0},
        {"loan_amount": 3.0, "monthly_income": 6.0},
    ]

    matrix = compute_correlation_matrix(feature_rows, ("loan_amount", "monthly_income"))

    assert matrix["loan_amount"]["loan_amount"] == 1.0
    assert matrix["monthly_income"]["monthly_income"] == 1.0
    assert round(matrix["loan_amount"]["monthly_income"], 6) == 1.0
