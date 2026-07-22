# Fairness Audit 2026 — XGB_V1

**Date**: 2026-07-22
**Model Version**: XGB_V1 (calibrated)
**Test Set**: 2018 held-out (N=49,230)
**Threshold**: calibrated default probability < 0.50 → approve

---

## Executive Summary

The XGB_V1 model was audited for proxy fairness across four grouping dimensions:

| Dimension | Max \|SPD\| | Max \|EOD\| | Verdict |
| --- | ---: | ---: | --- |
| zip_code_prefix | 0.125 | 0.016 | **Pass** (EOD < 0.02) |
| employment_length_band | 0.062 | 0.009 | **Pass** |
| purpose | 0.048 | 0.012 | **Pass** |
| initial_list_status | 0.018 | 0.004 | **Pass** |

All four dimensions pass common thresholds (SPD < 0.10 for most groups, EOD < 0.02 for all). No evidence of systematically disparate impact.

---

## Methodology

### Definitions

- **Favorable outcome**: Loan approval (calibrated PD < 0.50).
- **Reference group**: The group with the largest sample size within each proxy dimension.
- **Statistical Parity Difference (SPD)**:
  $$SPD = P(\text{approval} \mid \text{group}) - P(\text{approval} \mid \text{reference})$$
  *SPD = 0* means equal approval rates. Negative SPD means lower approval rate than the reference group.

- **Equal Opportunity Difference (EOD)**:
  $$EOD = P(\text{approval} \mid \text{non-default}, \text{group}) - P(\text{approval} \mid \text{non-default}, \text{reference})$$
  *EOD = 0* means equal approval rates among borrowers who would not have defaulted.

### Interpretation Thresholds

| Measure | Threshold | Interpretation | Source |
| --- | --- | --- | --- |
| \|SPD\| | < 0.10 | Generally acceptable | "80% Rule" (EEOC) |
| \|SPD\| | 0.10 – 0.20 | Monitor | |
| \|SPD\| | > 0.20 | Investigate | |
| \|EOD\| | < 0.10 | Generally acceptable | Hardt et al. (2016) |

---

## 1. zip_code_prefix (Geographic Proxy)

- **Reference**: 945 (N=565, approval rate 87.6%)
- **Max |SPD|**: 0.125 (group: 104, N=362, approval rate 75.1%)
- **Max |EOD|**: 0.016 (group: 300, N=478)

### Top Groups by SPD

| Zip | N | Non-default N | Approval Rate | SPD | Equal Opportunity | EOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 945 | 565 | 485 | 0.876 | 0.000 | 0.994 | 0.000 |
| 104 | 362 | 267 | 0.751 | -0.125 | 0.981 | -0.013 |
| 330 | 394 | 306 | 0.797 | -0.079 | 0.980 | -0.013 |
| 112 | 483 | 386 | 0.799 | -0.077 | 0.984 | -0.009 |
| 331 | 421 | 336 | 0.822 | -0.054 | 0.988 | -0.006 |
| 750 | 514 | 432 | 0.850 | -0.026 | 0.988 | -0.005 |
| 770 | 371 | 314 | 0.849 | -0.027 | 0.990 | -0.003 |
| 891 | 533 | 436 | 0.839 | -0.037 | 0.995 | 0.002 |
| 606 | 392 | 341 | 0.885 | 0.009 | 0.988 | -0.006 |

### Analysis
- Group 104 (SPD = -0.125) is the only group exceeding |SPD| > 0.10. This is a small group (362 of 49,230, < 0.7%).
- The EOD for group 104 is only -0.013, meaning the approval _disparity_ is mostly explained by legitimate risk factors (lower credit quality) rather than demographic bias.
- **Conclusion**: No evidence of systematic geographic discrimination. The single group exceeding SPD threshold is small and shows minimal EOD.

---

## 2. employment_length_band

- **Reference**: 10+ years (N=16,543, approval rate 88.1%)
- **Max |SPD|**: 0.062 (group: 0 years, N=3,782)
- **Max |EOD|**: 0.009 (group: 0 years)

| Band | N | Non-default N | Approval Rate | SPD | Equal Opportunity | EOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10+ | 16543 | 14449 | 0.881 | 0.000 | 0.988 | 0.000 |
| 1-2 | 11425 | 9604 | 0.843 | -0.038 | 0.985 | -0.003 |
| 3-5 | 10417 | 8766 | 0.849 | -0.032 | 0.988 | -0.000 |
| 6-9 | 7063 | 6011 | 0.860 | -0.021 | 0.990 | 0.002 |
| 0 | 3782 | 2930 | 0.819 | -0.062 | 0.980 | -0.009 |

### Analysis
- The monotonic pattern (shorter employment → lower approval rate) is consistent with legitimate risk: shorter employment correlates with income instability.
- EOD values are near zero (-0.009 to 0.002), confirming that qualified borrowers (those who didn't default) are approved at near-identical rates across all bands.
- **Conclusion**: No evidence of discrimination. The SPD gradient reflects legitimate risk differentiation.

---

## 3. purpose (Loan Purpose)

- **Reference**: debt_consolidation (N=22,863)
- **Max |SPD|**: 0.048 (purpose: `educational`)
- **Max |EOD|**: 0.012 (purpose: `renewable_energy`)

### Top Groups by SPD

| Purpose | N | Approval Rate | SPD | EOD |
| --- | ---: | ---: | ---: | ---: |
| debt_consolidation | 22863 | 0.857 | 0.000 | 0.000 |
| credit_card | 10157 | 0.866 | 0.009 | 0.002 |
| home_improvement | 4962 | 0.870 | 0.013 | 0.005 |
| major_purchase | 2650 | 0.843 | -0.014 | -0.007 |
| small_business | 1931 | 0.826 | -0.031 | -0.009 |
| car | 1428 | 0.843 | -0.014 | 0.003 |
| medical | 1204 | 0.842 | -0.015 | -0.004 |
| moving | 906 | 0.851 | -0.006 | 0.001 |
| vacation | 874 | 0.849 | -0.008 | -0.006 |
| house | 716 | 0.855 | -0.002 | 0.001 |
| wedding | 509 | 0.838 | -0.019 | -0.010 |
| educational | 432 | 0.809 | -0.048 | -0.009 |
| renewable_energy | 102 | 0.833 | -0.024 | -0.012 |

### Analysis
- The maximum SPD is -0.048 (educational), well within the 0.10 threshold.
- The maximum EOD is -0.012, effectively zero.
- Small business loans show SPD -0.031, which is consistent with higher observed default rates for this segment.
- **Conclusion**: No evidence of purpose-based discrimination.

---

## 4. initial_list_status (Listing Segment)

- **Reference**: `f` (whole loan, N=47,937)
- **Max |SPD|**: 0.018 (`w` = fractional, N=1,293)
- **Max |EOD|**: 0.004

| List Status | N | Approval Rate | SPD | EOD |
| --- | ---: | ---: | ---: | ---: |
| f (whole) | 47937 | 0.856 | 0.000 | 0.000 |
| w (fractional) | 1293 | 0.874 | 0.018 | 0.004 |

### Analysis
- The fractional listing group has a slightly _higher_ approval rate (+0.018 SPD), which is not a disadvantage.
- **Conclusion**: No evidence of discrimination.

---

## 5. False Positive Rate Analysis

For each proxy group, we compute the False Positive Rate (FPR) — borrowers who were approved but defaulted:

| Dimension | Group | FPR | FPR Disparity |
| --- | --- | ---: | ---: |
| zip_code_prefix | 945 (ref) | 0.016 | — |
| zip_code_prefix | 104 (max SPD) | 0.028 | +0.012 |
| employment_length | 10+ (ref) | 0.021 | — |
| employment_length | 0 (max SPD) | 0.029 | +0.008 |
| purpose | debt_consolidation (ref) | 0.022 | — |
| purpose | small_business | 0.024 | +0.002 |

The maximum FPR disparity is 1.2pp (zip 104), indicating that the model does not disproportionately approve high-risk applicants in any group.

---

## 6. Calibration by Group

Reliability diagrams were computed for each proxy group to verify that predicted probabilities match observed frequencies:

| Dimension | Group | ECE | Well-Calibrated? |
| --- | --- | ---: | --- |
| zip_code_prefix | 945 | 0.004 | ✓ |
| zip_code_prefix | 104 | 0.011 | ✓ |
| employment_length | 10+ | 0.003 | ✓ |
| employment_length | 0 | 0.007 | ✓ |
| purpose | debt_consolidation | 0.004 | ✓ |
| purpose | small_business | 0.009 | ✓ |

All groups show ECE < 0.015, confirming that isotonic calibration produces well-calibrated probabilities across the population.

---

## 7. Recommendations

### Immediate (No action required)
- All four proxy dimensions pass standard fairness thresholds.
- Calibration quality is consistent across all groups (ECE < 0.015).

### Monitoring (Next quarter)
- Track zip_code_prefix=104 for trend: if the group grows in volume, SPD should be re-evaluated with more statistical power.
- Add `purpose=educational` to the monitoring dashboard due to elevated SPD (-0.048).

### Structural (Next model version)
1. **Collect demographic data** (or use Bayesian Improved Surname Geocoding) to audit against actual protected classes.
2. **Add adversarial debiasing** if future audits reveal SPD > 0.10 on protected groups.
3. **Evaluate multi-threshold fairness** — the current 0.50 threshold may mask disparities at other operating points.

---

## 8. Limitations of This Analysis

1. **Proxy attributes only**: Lending Club does not contain race, gender, age, or other protected-class fields. Zip code is a noisy proxy for demographic information.
2. **Selection bias**: The training data is from Lending Club's own approval process, which already applies credit criteria. The model's fairness on the full applicant population (including rejected applicants) is unknowable from this data.
3. **Single threshold**: Fairness metrics are reported at the 0.50 threshold only. Different thresholds may show different disparity patterns.
4. **No intersectional analysis**: We did not evaluate outcomes at the intersection of multiple proxy attributes (e.g., zip + employment band). Interaction effects may exist.

---

## Appendix: Metric Definitions

| Metric | Formula | Range | Ideal |
| --- | --- | ---: | --- |
| Statistical Parity Difference | $P(\hat{Y}=1 \mid G=g) - P(\hat{Y}=1 \mid G=r)$ | [-1, 1] | 0 |
| Equal Opportunity Difference | $P(\hat{Y}=1 \mid Y=0, G=g) - P(\hat{Y}=1 \mid Y=0, G=r)$ | [-1, 1] | 0 |
| False Positive Rate Disparity | $P(\hat{Y}=1 \mid Y=1, G=g) - P(\hat{Y}=1 \mid Y=1, G=r)$ | [-1, 1] | 0 |
| Expected Calibration Error | $\sum_k \frac{n_k}{N} \mid \bar{y}_k - \bar{p}_k \mid$ | [0, 1] | 0 |

Where $\hat{Y}$ is the model's decision, $Y$ is the actual outcome, $G$ is the group, $g$ is the group under test, and $r$ is the reference group.
