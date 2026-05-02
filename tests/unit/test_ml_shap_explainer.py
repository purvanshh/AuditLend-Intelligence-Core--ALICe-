from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

from ml.data.features import build_feature_row
from ml.explain.shap_explainer import explain_feature_row
from ml.models.train import build_model_feature_names, encode_feature_row, infer_categories


def test_explain_feature_row_returns_ranked_contributions(tmp_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression

    feature_rows = [
        _build_feature_row("100", 18.0, 660.0, "RENT", 1),
        _build_feature_row("101", 12.0, 720.0, "MORTGAGE", 0),
        _build_feature_row("102", 27.0, 640.0, "RENT", 1),
        _build_feature_row("103", 9.0, 760.0, "OWN", 0),
        _build_feature_row("104", 22.0, 680.0, "RENT", 1),
        _build_feature_row("105", 8.0, 780.0, "MORTGAGE", 0),
    ]

    categories_by_feature = infer_categories(feature_rows)
    feature_names = build_model_feature_names(categories_by_feature)
    X = [encode_feature_row(row, categories_by_feature) for row in feature_rows]
    y = [int(row["target_defaulted"]) for row in feature_rows]

    model = RandomForestClassifier(n_estimators=24, max_depth=4, random_state=42)
    try:
        import pandas as pd
    except ImportError:
        model.fit(X, y)
    else:
        model.fit(pd.DataFrame(X, columns=feature_names), y)

    artifact_dir = tmp_path / "artifacts" / "XGB_V1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "random_forest.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    with (artifact_dir / "isotonic_calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "XGB_V1",
                "selected_candidate": "random_forest",
                "artifact_path": str(artifact_path),
                "feature_names": feature_names,
                "categories_by_feature": categories_by_feature,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    explanation = explain_feature_row(feature_rows[0], manifest_path, max_features=5)
    payload = explanation.to_audit_payload()

    assert explanation.model_version == "XGB_V1"
    assert explanation.calibrated_default_probability is not None
    assert explanation.model_factor_contributions
    assert payload["explanation_method"] == "shap"
    assert payload["model_summary"].startswith("Model factors:")
    assert len(payload["model_factor_contributions"]) == 5
    assert payload["model_factor_contributions"][0]["raw_value"] != ""

    absolute_contributions = [
        abs(float(row["shap_contribution"])) for row in payload["model_factor_contributions"]
    ]
    assert absolute_contributions == sorted(absolute_contributions, reverse=True)


def _build_feature_row(
    loan_id: str,
    dti_pct: float,
    fico_midpoint: float,
    home_ownership: str,
    defaulted: int,
) -> dict[str, object]:
    return build_feature_row(
        {
            "loan_id": loan_id,
            "issue_date": date(2018, 1, 1),
            "loan_status": "Charged Off" if defaulted else "Fully Paid",
            "grade": "C" if defaulted else "A",
            "sub_grade": "C4" if defaulted else "A2",
            "purpose": "debt_consolidation" if defaulted else "credit_card",
            "home_ownership": home_ownership,
            "verification_status": "Verified" if defaulted else "Source Verified",
            "loan_amount": 18000.0 if defaulted else 12000.0,
            "funded_amount": 18000.0 if defaulted else 12000.0,
            "term_months": 36,
            "interest_rate_pct": 17.5 if defaulted else 8.9,
            "installment": 640.0 if defaulted else 380.0,
            "monthly_income": 7000.0 if defaulted else 11000.0,
            "estimated_existing_emi": 2100.0 if defaulted else 700.0,
            "dti_pct": dti_pct,
            "fico_midpoint": fico_midpoint,
            "last_fico_midpoint": fico_midpoint + (10.0 if not defaulted else -20.0),
            "employment_length_years": 4.0,
            "earliest_credit_line": date(2008, 1, 1),
            "revol_util_pct": 78.0 if defaulted else 26.0,
            "bc_util_pct": 82.0 if defaulted else 30.0,
            "all_util_pct": 66.0 if defaulted else 24.0,
            "il_util_pct": 50.0 if defaulted else 18.0,
            "revol_bal": 42000.0 if defaulted else 9000.0,
            "tot_cur_bal": 88000.0 if defaulted else 36000.0,
            "total_bal_ex_mort": 54000.0 if defaulted else 14000.0,
            "total_rev_hi_lim": 48000.0 if defaulted else 30000.0,
            "total_bc_limit": 21000.0 if defaulted else 25000.0,
            "delinq_2yrs": 2.0 if defaulted else 0.0,
            "inq_last_6mths": 4.0 if defaulted else 1.0,
            "inq_last_12m": 6.0 if defaulted else 2.0,
            "open_acc": 8.0,
            "total_acc": 18.0,
            "mort_acc": 1.0 if home_ownership == "MORTGAGE" else 0.0,
            "pub_rec_bankruptcies": 1.0 if defaulted else 0.0,
            "tax_liens": 0.0,
            "percent_bc_gt_75": 55.0 if defaulted else 8.0,
            "pct_tl_nvr_dlq": 81.0 if defaulted else 98.0,
            "collections_12_mths_ex_med": 2.0 if defaulted else 0.0,
            "mo_sin_rcnt_rev_tl_op": 3.0 if defaulted else 12.0,
            "mo_sin_old_rev_tl_op": 90.0 if defaulted else 155.0,
            "open_rv_24m": 3.0,
            "open_il_24m": 1.0,
            "defaulted": defaulted,
        }
    )
