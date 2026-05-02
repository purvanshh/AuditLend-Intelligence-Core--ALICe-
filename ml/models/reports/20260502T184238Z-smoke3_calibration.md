# Phase 5 Calibration Report

Run ID: `20260502T184238Z-smoke3`
Selected candidate: `lightgbm`
Calibrator artifact: `ml/models/artifacts/20260502T184238Z-smoke3/isotonic_calibrator.pkl`

## Reliability Curves

- Validation curve: `ml/models/reports/20260502T184238Z-smoke3_validation_calibration.svg`
- Test curve: `ml/models/reports/20260502T184238Z-smoke3_test_calibration.svg`

## Before vs After

| Split | Stage | AUC-ROC | AUC-PR | Brier | ECE | Max Gap | Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | raw | 1.000000 | 1.000000 | 0.000000 | 0.000049 | 0.000100 | 200 |
| validation | calibrated | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 200 |
| test | raw | 0.975706 | 0.944672 | 0.026101 | 0.027782 | 0.721821 | 200 |
| test | calibrated | 0.959718 | 0.937190 | 0.026105 | 0.027866 | 0.721901 | 200 |

## Calibration Interpretation

- Validation calibration is fit using isotonic regression on the validation split probabilities.
- Test metrics show how that calibration transfers to unseen examples from the held-out test split.
- A well-calibrated model should keep reliability points close to the diagonal where predicted default probability matches observed default frequency.
