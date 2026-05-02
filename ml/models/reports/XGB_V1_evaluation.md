# Official XGB_V1 Evaluation Report

Model version: `XGB_V1`
Manifest: `ml/models/manifest.yaml`

## Split Metrics

| Split | Variant | AUC-ROC | AUC-PR | Brier | ECE | Max Gap | Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | raw | 0.949989 | 0.824472 | 0.068222 | 0.003109 | 0.011656 | 1116769 |
| train | calibrated | 0.949897 | 0.824723 | 0.070676 | 0.025633 | 0.138908 | 1116769 |
| validation | raw | 0.980275 | 0.942174 | 0.037116 | 0.015861 | 0.089809 | 156290 |
| validation | calibrated | 0.980246 | 0.942077 | 0.036385 | 0.004349 | 0.092811 | 156290 |
| test | raw | 0.975786 | 0.936718 | 0.026582 | 0.016177 | 0.158911 | 49230 |
| test | calibrated | 0.975664 | 0.936609 | 0.025293 | 0.003550 | 0.072691 | 49230 |

## Selected Parameters

`{"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "n_estimators": 200, "reg_lambda": 1.0, "subsample": 0.8}`

## Top Feature Importance

| Feature | Importance |
| --- | ---: |
| credit_score_recent_delta | 0.35809827 |
| grade_A | 0.12332106 |
| grade_B | 0.06082303 |
| term_months | 0.05801700 |
| interest_rate_pct | 0.04164733 |
| grade_C | 0.03085880 |
| credit_score_midpoint | 0.02582905 |
| all_util_ratio | 0.02547945 |
| open_revolving_24m | 0.02186508 |
| grade_D | 0.02140900 |
| il_util_ratio | 0.01268713 |
| loan_amount_to_income | 0.01119503 |
| funded_amount | 0.01093631 |
| existing_emi_to_income | 0.00899617 |
| loan_amount | 0.00854035 |
