# WOE / IV Analysis Report

*Generated: 2026-07-22 | Data: Lending Club (2007–2018)*

## Methodology

**Weight of Evidence (WOE)** measures the log-odds of non-defaults vs defaults in each bin:

$$WOE_i = \ln\left(\frac{\%\ \text{non-defaults in bin}_i}{\%\ \text{defaults in bin}_i}\right)$$

**Information Value (IV)** aggregates WOE across bins:

$$IV = \sum_i (\%\ \text{non-defaults}_i - \%\ \text{defaults}_i) \times WOE_i$$

### Predictive Power Scale

| IV Range | Interpretation |
| --- | --- |
| < 0.02 | Not useful for modeling |
| 0.02 – 0.10 | Weak predictor |
| 0.10 – 0.30 | Medium predictor |
| > 0.30 | **Strong predictor** |

## Results

| Feature | IV | Power |
| --- | ---: | --- |
| dti_ratio | 0.3842 | **Strong** |
| interest_rate_pct | 0.3418 | **Strong** |
| revol_util_ratio | 0.2156 | Medium |
| credit_score_midpoint | 0.1873 | Medium |
| all_util_ratio | 0.1567 | Medium |
| loan_amount_to_income | 0.1421 | Medium |
| existing_emi_to_income | 0.1289 | Medium |
| bc_util_ratio | 0.1145 | Medium |
| credit_history_age_years | 0.0894 | Weak |
| employment_length_years | 0.0345 | Weak |
| monthly_income | 0.0189 | Not useful |

## Feature-by-Feature Analysis

### dti_ratio (IV = 0.3842 — **Strong**)

Debt-to-income ratio is the single most powerful predictor of default. Borrowers with DTI > 40% show sharply negative WOE, indicating default risk 2–3× higher than the population average. This is consistent with decades of credit industry research: debt burden is the primary driver of consumer credit risk.

**WOE Pattern**: Monotonically decreasing — as DTI increases, the proportion of non-defaults falls linearly. This monotonicity makes DTI ideal for logistic regression and scorecard modeling without transformation.

**Business rule**: The existing 35% DTI threshold in RULE_SET_V1 is well-calibrated.

### interest_rate_pct (IV = 0.3418 — **Strong**)

Interest rate embeds the platform's own risk assessment (pricing for risk). Rates above 15% correspond to borrowers with significantly elevated default probability. The IV is high partly because rate is assigned based on the same risk factors the model learns.

**WOE Pattern**: Strongly monotonic — higher rate → lower WOE. No crossing or non-monotonic bins.

**Caveat**: Interest rate is partly a function of the dependent variable (risk-based pricing). Its predictive power may not fully generalize to portfolios with different pricing strategies.

### revol_util_ratio (IV = 0.2156 — **Medium**)

Revolving utilization captures liquidity stress. Utilization > 80% corresponds to borrowers with limited financial headroom. The WOE curve is approximately monotonic with a steeper drop above 70% utilization.

**Key insight**: Utilization provides signal beyond DTI — a borrower can have low DTI but high utilization, indicating different risk profiles.

### credit_score_midpoint (IV = 0.1873 — **Medium**)

FICO scores are predictive but partially redundant with grade (the platform's letter grade is derived from similar input data). Borrowers with scores below 660 show substantially elevated default risk.

**Note**: FICO's predictive power in this dataset is lower than typical because Lending Club's own grade assignment already captures much of the same information.

### monthly_income (IV = 0.0189 — **Not useful**)

Raw income, without context, is almost non-predictive. A borrower earning ₹50L/month can still default if their DTI is 60%. This underscores the importance of ratio-based features over raw amounts.

## Key Takeaways

1. **DTI is king** — IV > 0.30, strongly monotonic, and theoretically grounded. It should remain the primary gating factor in any scorecard.

2. **Interest rate captures embedded risk** — High IV but partially endogenous. Models should include it but also evaluate performance on pricing-adjusted metrics.

3. **Ratios > raw amounts** — DTI, loan-to-income, utilization ratios all outperform raw income, loan amount, and credit score alone.

4. **Utilization adds orthogonal signal** — Beyond DTI, utilization captures working-capital stress that DTI misses.

5. **Employment length is weak** — Despite common industry use, employment length (IV = 0.03) adds little predictive value beyond other features.

## Business Implications

- **Scorecard design**: Use WOE-transformed DTI, utilization, and interest rate for logistic regression scorecards.
- **Feature engineering priority**: Focus on ratio features (to income) — not raw amounts.
- **Underwriting policy**: DTI gates should remain strict. Utilization monitoring adds incremental value.
- **Verification priority**: Verifying DTI (both declared and bank-derived) is more valuable than verifying raw income.

## Appendix: Monotonicity Check

| Feature | Monotonic WOE? | Suitable for Logistic Regression? |
| --- | --- | --- |
| dti_ratio | Yes | ✓ Directly |
| interest_rate_pct | Yes | ✓ Directly |
| revol_util_ratio | Near-monotonic | ✓ With binning |
| credit_score_midpoint | Yes | ✓ Directly |
| all_util_ratio | Near-monotonic | ✓ With binning |
| loan_amount_to_income | Near-monotonic | ✓ With binning |
| existing_emi_to_income | Near-monotonic | ✓ With binning |
| credit_history_age_years | Partial | ⚠ Needs fine binning |
| employment_length_years | Partial | ⚠ Needs fine binning |
| monthly_income | No | ✗ Not recommended |

*Monotonicity determined by visual inspection of WOE bar plots. "Near-monotonic" allows at most one direction reversal between adjacent bins.*
