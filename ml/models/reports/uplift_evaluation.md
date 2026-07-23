# Uplift Model Evaluation

## Summary
- Two-model uplift (XGBoost) trained on {n} matched pairs
- Qini coefficient: {qini:.3f}
- Identified {pct_responsive:.1f}% of applicants as "uplift-responsive" (approval reduces default risk)

## Methodology
- Treatment model: XGBoost on approved loans
- Control model: XGBoost on propensity-matched declined loans
- Uplift score = P(default|treatment) - P(default|control)
- Negative uplift → approval reduces default risk
- Positive uplift → applicant likely to default regardless of approval

### Two-Model Approach
The two-model approach trains separate classifiers for the treatment group (approved loans)
and the control group (declined loans). The uplift score for each applicant is computed as
the difference in predicted default probabilities between the two models:

```
uplift = P_treatment(default = 1 | X) - P_control(default = 1 | X)
```

### Propensity Score Matching
Control observations are selected via nearest-neighbor propensity score matching (PSM) on
a logistic regression classifier trained to predict approval probability. This reduces
selection bias by ensuring treatment and control groups have similar observed characteristics.

## Data Requirements
- **Treatment group**: Loans that were approved, with known default outcomes
- **Control group**: Propensity-matched declined loans with similar observable characteristics
- **Minimum requirements**: At least 500 treatment and 500 control observations per segment
- **Feature alignment**: All features must be present in both groups; categorical variables
  must use identical encoding

## Evaluation Metrics

### Qini Coefficient
The Qini coefficient measures the area between the model's uplift curve and the diagonal
(random sorting). It quantifies how well the model separates responsive from non-responsive
applicants:

- Qini > 0: Model successfully identifies responsive segments (better than random)
- Qini = 0: Model performs no better than random sorting
- Qini < 0: Model is worse than random

The Qini curve plots cumulative uplift against the proportion of the population targeted,
ordering applicants by predicted uplift from highest to lowest.

### Uplift Curve
The uplift curve shows the incremental number of defaults prevented when targeting a given
fraction of the population, compared to random selection. The area under this curve relative
to the diagonal gives the Qini coefficient.

## Segment Analysis
| Segment | Size | Avg Uplift | Default Rate (Treatment) | Default Rate (Control) | Recommendation |
| --- | ---: | ---: | ---: | ---: | --- |
| High DTI, Low Credit Score | ... | ... | ... | ... | Decline |
| Low DTI, High Credit Score | ... | ... | ... | ... | Approve |
| Medium DTI, Medium Credit Score | ... | ... | ... | ... | Standard Review |

### Interpretation
- **Negative uplift segments**: These applicants perform better when approved than when
  declined. Approval is beneficial — it reduces their default probability.
- **Near-zero uplift segments**: Approval has little causal effect on default. Standard
  risk assessment should be used.
- **Positive uplift segments**: These applicants would default regardless of approval.
  Declining is recommended as the loan does not improve outcomes.

## Policy Implications

### Uplift-Based Decision Rules
1. **Uplift < -0.05**: Strongly recommend approval
   - Approval reduces default risk
   - These applicants are genuinely creditworthy

2. **-0.05 <= Uplift <= 0.05**: Use standard risk assessment
   - Neutral uplift; no strong causal effect
   - Defer to existing scorecard and rules

3. **Uplift > 0.05**: Recommend decline
   - Applicant likely to default regardless of approval
   - Loan approval does not improve outcomes

### Integration with Existing Pipeline
The uplift model runs alongside the existing credit risk model in the decision pipeline:

```
Applicant Features
    |
    +---> Risk Score (existing model) ---> risk_score, confidence
    |
    +---> Uplift Score (uplift model) ---> uplift_score, recommendation
    |
    v
Decision Engine
    |
    +---> APPROVE (if risk low AND uplift negative)
    +---> DECLINE (if risk high OR uplift positive)
    +---> MANUAL_REVIEW (if conflicting signals)
```

The uplift score is exposed as an additional factor in the explanation output, allowing
underwriters to understand not just whether an applicant is likely to default, but whether
approval would change that outcome.

### Monitoring
- Track Qini coefficient over time to detect degradation
- Monitor segment-level uplift distributions for concept drift
- Re-run propensity score matching periodically as population shifts
- A/B test uplift-based decisions against standard decisions for neutral segments
