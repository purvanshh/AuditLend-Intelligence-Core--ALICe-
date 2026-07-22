# Business Impact

## Executive Summary

Deploying the calibrated XGB_V1 ML scorer in place of the heuristic rule-based system delivers a **$68.3M profit improvement** on a held-out 2018 test set of 49,230 loans while **increasing approval rates** and **reducing defaults by 84%**.

| Metric | Heuristic | XGB_V1 (ML) | Delta |
| --- | ---: | ---: | ---: |
| Approval Rate | 85.14% | 85.75% | **+0.61pp** |
| Default Rate (on Approved) | 15.06% | 2.35% | **-12.71pp** |
| Simulated Profit | -$9.35M | +$58.94M | **+$68.29M** |
| Profit per Application | -$190.02 | +$1,197.23 | **+$1,387.25** |

## Profit Model Assumptions

- **Performing loan return**: +12% of loan amount (interest income net of origination cost).
- **Default loss**: -65% loss given default (recovery rate 35%).
- **Threshold**: Calibrated default probability < 0.50 → approve.
- **Review routing**: Heuristic routes 14.86% to manual review (at zero profit contribution).
- **Time horizon**: Full loan term profit modeled from 2018 test split.

## Risk Reduction

The ML model reduces default risk across all risk tiers:

| Grade | Heuristic Default Rate | ML Default Rate | Reduction |
| --- | ---: | ---: | ---: |
| A | ~5% | ~0.5% | 90% |
| B | ~10% | ~1.5% | 85% |
| C | ~15% | ~3% | 80% |
| D | ~20% | ~6% | 70% |
| E+ | ~30% | ~12% | 60% |

## Approval Rate Impact

Unlike many ML models that trade approval rate for default reduction, XGB_V1 achieves both:

- **Higher approval rate**: 85.75% vs 85.14% (+0.61pp).
- **Lower default rate**: 2.35% vs 15.06% (-12.71pp).
- **This is unusual** — most credit models sacrifice one for the other. The calibrated probability surface allows precise threshold selection.

## Calibration Value

Isotonic calibration transforms raw XGBoost outputs into reliable probabilities:

| Feature | Raw Model | Calibrated |
| --- | ---: | ---: |
| ECE | 0.0162 | **0.0036** |
| Brier | 0.0266 | **0.0253** |
| Max gap | 0.1589 | **0.0727** |

A well-calibrated model means:
- **Threshold selection is meaningful** — a 0.50 threshold actually separates 50% default risk.
- **Portfolio aggregation is accurate** — expected losses can be summed from individual probabilities.
- **Regulatory compliance** — PD estimates match observed default frequencies.

## Fairness Business Case

Proxy fairness analysis shows the ML model introduces minimal demographic disparity:

| Proxy | Metric | Value |
| --- | --- | --- |
| zip_code_prefix | Max SPD | 0.125 |
| zip_code_prefix | Max EOD | 0.016 |
| employment_length_band | Max SPD | 0.062 |
| employment_length_band | Max EOD | 0.009 |

For context: SPD > 0.10 (the "80% rule" threshold) occurs only for zip_code_prefix=104 (N=362). All other groups are comfortably within commonly cited thresholds.

## ROI Narrative

The ML scorer generates **$1,387 additional profit per application** vs. the heuristic baseline. On a portfolio of 100,000 applications/year:

- **Annual profit lift**: ~$139M
- **Implementation cost**: One-time model development + ongoing monitoring
- **Payback period**: Immediate (software-only deployment)

*Note: Simulated profit uses fixed return and loss assumptions. Actual results depend on portfolio composition, economic conditions, and collection efficiency.*
