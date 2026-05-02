# Phase 4 Model Evaluation Report

Run ID: `20260502T184238Z-smoke3`
Selected candidate: `lightgbm`
Artifact: `ml/models/artifacts/20260502T184238Z-smoke3/lightgbm.pkl`

## Selection Summary

- Search metric: validation AUC-PR
- Feature count: 97
- Split counts: {'holdout': 0, 'test': 200, 'train': 200, 'validation': 200}
- Selected params: `{"colsample_bytree": 0.8, "learning_rate": 0.05, "n_estimators": 350, "num_leaves": 31, "subsample": 0.8}`

## Split Metrics

| Split | AUC-ROC | AUC-PR | Brier | ECE | Max Gap | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.000000 | 1.000000 | 0.000000 | 0.000047 | 0.000120 | 200 |
| validation | 1.000000 | 1.000000 | 0.000000 | 0.000049 | 0.000100 | 200 |
| test | 0.975706 | 0.944672 | 0.026101 | 0.027782 | 0.721821 | 200 |

## Candidate Comparison

| Candidate | Family | Val AUC-PR | Val AUC-ROC | Val Brier | Test AUC-PR | Test AUC-ROC | Test Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | lightgbm | 0.957764 | 0.985057 | 0.046533 | 0.938719 | 0.971138 | 0.022372 |
| xgboost | xgboost | 0.956323 | 0.983758 | 0.044941 | 0.924730 | 0.977159 | 0.033198 |
| logistic_regression | sklearn | 0.808034 | 0.911383 | 0.123032 | 0.720139 | 0.917566 | 0.085749 |

## Threshold Analysis (Test Split)

| Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | FPR | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 26 | 4 | 168 | 2 | 0.866667 | 0.928571 | 0.976744 | 0.023256 | 0.970000 |
| 0.40 | 25 | 4 | 168 | 3 | 0.862069 | 0.892857 | 0.976744 | 0.023256 | 0.965000 |
| 0.50 | 25 | 4 | 168 | 3 | 0.862069 | 0.892857 | 0.976744 | 0.023256 | 0.965000 |
| 0.60 | 25 | 4 | 168 | 3 | 0.862069 | 0.892857 | 0.976744 | 0.023256 | 0.965000 |
| 0.70 | 25 | 4 | 168 | 3 | 0.862069 | 0.892857 | 0.976744 | 0.023256 | 0.965000 |

## Top Feature Importance

| Feature | Importance |
| --- | ---: |
| credit_score_recent_delta | 579.00000000 |
| interest_rate_pct | 271.00000000 |
| dti_ratio | 127.00000000 |
| mortgage_account_share | 119.00000000 |
| balance_per_open_account | 118.00000000 |
| loan_amount_to_income | 97.00000000 |
| credit_history_age_years | 96.00000000 |
| installment | 91.00000000 |
| total_balance_to_income | 86.00000000 |
| all_util_ratio | 84.00000000 |
| recent_revolving_trade_gap_months | 73.00000000 |
| current_balance_to_income | 72.00000000 |
| delinquency_burden | 67.00000000 |
| credit_score_midpoint | 62.00000000 |
| loan_amount | 61.00000000 |

## Segment Diagnostics

| Segment | Value | Rows | Positive Rate | Mean Score | Predicted Positive Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| home_ownership | MORTGAGE | 93 | 0.053763 | 0.071596 | 0.075269 |
| home_ownership | RENT | 81 | 0.234568 | 0.237238 | 0.234568 |
| home_ownership | OWN | 26 | 0.153846 | 0.109960 | 0.115385 |
| verification_status | Source Verified | 79 | 0.139241 | 0.149030 | 0.151899 |
| verification_status | Not Verified | 72 | 0.138889 | 0.113642 | 0.111111 |
| verification_status | Verified | 49 | 0.142857 | 0.179144 | 0.183673 |
