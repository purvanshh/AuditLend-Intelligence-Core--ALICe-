# Model Card: XGB_V1

## Model Overview

- **Model Name**: XGB_V1
- **Model Version**: 1.0.0
- **Model Type**: Gradient-boosted decision tree (XGBoost) with isotonic calibration
- **Date**: 2026-05-02
- **Maintainer**: AuditLend ML Platform
- **License**: Internal use — not for redistribution

## Intended Use

### Primary Use
Credit default risk scoring for unsecured consumer loan applications. The model estimates probability of default (PD) over the full loan term, which is used to:

- Determine accept/decline decisions (threshold: calibrated PD < 0.50).
- Route ambiguous cases to manual review (via calibrated confidence < 0.60).
- Provide per-prediction factor explanations via SHAP values.
- Enable A/B comparison against the heuristic rule-based scorecard.

### Primary Users
- Automated credit decision engine (API consumers).
- Credit risk analysts reviewing model outputs.
- Compliance teams auditing decision fairness.

### Out-of-Scope Uses
- **Secured loans**: The model was trained on unsecured consumer loans only.
- **Business lending**: Training data is individual borrower loans only.
- **Real-time fraud detection**: The model predicts credit risk, not fraud.
- **Regulatory capital calculation**: PD estimates have not been validated under Basel/IFRS 9 requirements.
- **Adversarial settings**: No adversarial robustness testing has been performed.

## Training Data

### Source
Lending Club accepted loan data from 2007 Q2 through 2018 Q4.

### Filtering
- Individual applications only (joint/co-applicant excluded).
- Terminal loan statuses only: `Fully Paid` (non-default) and `Charged Off` (default).
- Current, In Grace Period, and Late statuses excluded.
- Rows with missing or zero income, loan amount, or term excluded.

### Size and Split

| Split | Period | Rows |
| --- | --- | ---: |
| Train | 2007–2016 | 1,116,769 |
| Validation | 2017 | 156,290 |
| Test | 2018 | 49,230 |

### Label Definition
- **Default (positive class, 1)**: `Charged Off` or `Does not meet credit policy. Status:Charged Off`.
- **Non-default (negative class, 0)**: `Fully Paid` or `Does not meet credit policy. Status:Fully Paid`.
- Default rate in training data: ~15.2%.

### Features
- **38 engineered features** from raw Lending Club fields.
- Top 5 by importance: `credit_score_recent_delta` (0.36), `grade_A` (0.12), `grade_B` (0.06), `term_months` (0.06), `interest_rate_pct` (0.04).
- Categorical features: `grade` (A–G), `sub_grade`, `purpose`, `home_ownership`, `verification_status`.

## Model Architecture

### Algorithm
- **Base model**: XGBoost
- **Hyperparameters**: `colsample_bytree=0.8`, `learning_rate=0.05`, `max_depth=6`, `min_child_weight=5`, `n_estimators=200`, `reg_lambda=1.0`, `subsample=0.8`
- **Calibration**: Isotonic regression (fit on validation split)
- **Explainability**: SHAP TreeExplainer (top-5 features per prediction)
- **Drift detection**: Kolmogorov-Smirnov test against training reference snapshot

### Training Details
- **Search**: 36 candidate hyperparameter combinations evaluated.
- **Selection criterion**: Validation AUC-PR (0.9386 calibrated).
- **Hardware**: Local workstation.
- **Training time**: ~15 minutes (full corpus).

## Performance

### Held-Out Test Metrics (2018, N=49,230)

| Metric | Raw Model | Calibrated |
| --- | ---: | ---: |
| AUC-ROC | 0.9758 | **0.9757** |
| AUC-PR | 0.9367 | **0.9366** |
| Brier Score | 0.0266 | **0.0253** |
| ECE | 0.0162 | **0.0036** |
| Max Calibration Gap | 0.1589 | **0.0727** |

### Business Impact (Heuristic vs. ML, threshold 0.50)

| Arm | Approval Rate | Default Rate (Approved) | Simulated Profit |
| --- | ---: | ---: | ---: |
| Heuristic | 85.14% | 15.06% | -$9.35M |
| XGB_V1 | 85.75% | **2.35%** | **+$58.94M** |
| **Delta** | **+0.61pp** | **-12.71pp** | **+$68.29M** |

### Calibration Quality
- Isotonic calibration reduced ECE by **78%** (0.016 → 0.0036).
- Reliability diagrams confirm calibration holds across all risk deciles.
- Per-group calibration verified for zip_code_prefix and employment_length_band.

## Fairness Analysis

### Methodology
- **Proxy attributes**: `zip_code_prefix` (geographic proxy) and `employment_length_band`.
- **Favorable outcome**: Loan approval (calibrated PD < 0.50).
- **Equal opportunity**: Measured on the non-default class (favorable repayment outcome).

### Results

| Proxy Attribute | Reference | Max \|SPD\| | Max \|EOD\| |
| --- | --- | ---: | ---: |
| zip_code_prefix | 945 | **0.125** | **0.016** |
| employment_length_band | 10+ years | **0.062** | **0.009** |

- Largest SPD gap: `zip_code_prefix=104` (SPD = -0.125). This is a small group (N=362).
- All EOD values are below 0.02, indicating near-equal opportunity across groups.
- These are proxy-based diagnostics, not protected-class measurements.

## Limitations

1. **Proxy fairness only**: Lending Club does not contain protected class data. Real fairness auditing requires demographic data.
2. **No economic cycle modeling**: Training data (2007–2018) includes the 2008 recession and 2010s expansion, but the model does not explicitly condition on macro variables.
3. **Feature proxy quality**: Live inference uses conservative proxies for fields not in the application schema (e.g., revolving trade history). This reduces absolute performance vs. full-schema scoring.
4. **No COVID-like shock**: The training data predates COVID-19. Model behavior under pandemic-level economic stress is untested.
5. **Static threshold**: The 0.50 default probability threshold is fixed. Optimal thresholds may vary with economic conditions and portfolio strategy.

## Ethical Considerations

- **Transparency**: Every decision includes SHAP-based factor contributions, enabling borrowers and reviewers to understand why a decision was made.
- **Appeal mechanism**: Manual review routing ensures no automated decision is final without human oversight.
- **Monitoring**: Live drift detection and fairness metrics are tracked via Prometheus. Alerts trigger investigation.
- **Data privacy**: Raw PII is never used in model training or inference. All PII is AES-256-GCM encrypted at rest.

## Maintenance

| Activity | Frequency | Owner |
| --- | --- | --- |
| Drift monitoring | Per-inference | ML Platform |
| Fairness audit | Quarterly | Risk Analytics |
| Full retraining | Annual or upon drift alert | ML Platform |
| Model card review | Semi-annual | Governance Committee |
| Threshold calibration | Upon strategy change | Credit Risk |

## Caveats and Recommendations

1. **Do not use for secured lending** without retraining on appropriate data.
2. **Review proxy mappings** before deploying to a new geographic market.
3. **Validate calibration** on the target population before production use.
4. **Monitor DTI and interest_rate_pct drift** — these are the most sensitive features.
5. **Consider ensemble** with the survival model (Cox PH) for time-sensitive decisions.

---

*This model card follows the framework described in Mitchell et al. (2019), "Model Cards for Model Reporting."*
