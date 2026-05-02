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

## Fairness Reference Analysis

Approval is treated as the favorable outcome in this reference analysis.

### `zip_code_prefix`

- Threshold: `0.50`
- Reference group: `945` (565 rows)
- Reference approval rate: `0.876106`
- Reference equal opportunity: `0.993814`
- Max |statistical parity difference|: `0.124725`
- Max |equal opportunity difference|: `0.015766`

| Group | Rows | Non-default Rows | Approval Rate | SPD | Equal Opportunity | EOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 945 | 565 | 485 | 0.876106 | 0.000000 | 0.993814 | 0.000000 |
| 891 | 533 | 436 | 0.838649 | -0.037457 | 0.995413 | 0.001598 |
| 750 | 514 | 432 | 0.850195 | -0.025912 | 0.988426 | -0.005389 |
| 112 | 483 | 386 | 0.799172 | -0.076934 | 0.984456 | -0.009358 |
| 300 | 478 | 410 | 0.859833 | -0.016274 | 0.978049 | -0.015766 |
| 331 | 421 | 336 | 0.821853 | -0.054253 | 0.988095 | -0.005719 |
| 330 | 394 | 306 | 0.796954 | -0.079152 | 0.980392 | -0.013422 |
| 606 | 392 | 341 | 0.885204 | 0.009098 | 0.988270 | -0.005545 |
| 770 | 371 | 314 | 0.849057 | -0.027050 | 0.990446 | -0.003369 |
| 917 | 364 | 318 | 0.876374 | 0.000267 | 0.981132 | -0.012682 |
| 104 | 362 | 267 | 0.751381 | -0.124725 | 0.981273 | -0.012541 |
| 070 | 356 | 306 | 0.856742 | -0.019365 | 0.986928 | -0.006886 |

### `employment_length_band`

- Threshold: `0.50`
- Reference group: `10+` (16543 rows)
- Reference approval rate: `0.880796`
- Reference equal opportunity: `0.988096`
- Max |statistical parity difference|: `0.061652`
- Max |equal opportunity difference|: `0.008574`

| Group | Rows | Non-default Rows | Approval Rate | SPD | Equal Opportunity | EOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10+ | 16543 | 14449 | 0.880796 | 0.000000 | 0.988096 | 0.000000 |
| 1-2 | 11425 | 9604 | 0.842888 | -0.037907 | 0.985423 | -0.002673 |
| 3-5 | 10417 | 8766 | 0.849189 | -0.031607 | 0.988022 | -0.000074 |
| 6-9 | 7063 | 6011 | 0.859550 | -0.021246 | 0.990185 | 0.002089 |
| 0 | 3782 | 2930 | 0.819143 | -0.061652 | 0.979522 | -0.008574 |
