# Survival Analysis for Credit Risk

**Why survival analysis?** Binary classification (default vs. paid) answers *if* someone defaults. Survival analysis answers *when* — and *when* matters for provisioning, pricing, and portfolio stress testing.

## Methods Implemented

### Kaplan-Meier Estimator (`ml/models/survival_km.py`)

Non-parametric estimator of the survival function:

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

- Greenwood confidence intervals for uncertainty quantification.
- Stratified analysis by `grade`, `purpose`, `verification_status`.
- Median survival time and survival-at-time-t queries.

**Usage:**

```python
from ml.models.survival_km import KaplanMeierFitter

fitter = KaplanMeierFitter()
result = fitter.fit(durations, events)
curve = result.overall_curve

# Stratified by grade
strata = fitter.fit_stratified(feature_rows, strata_col='grade')
for grade, curve in strata.strata.items():
    print(f"Grade {grade}: 12-month survival = {curve.survival_at_time(12):.3f}")
```

### Cox Proportional Hazards (`ml/models/survival_coxph.py`)

Semi-parametric model of the hazard function:

$$h(t|X) = h_0(t) \cdot \exp(\beta_1 X_1 + \dots + \beta_p X_p)$$

- Newton-Raphson optimization of partial likelihood.
- Breslow approximation for tied event times.
- Output: coefficients, hazard ratios (exp(β)), standard errors, z-scores, p-values.
- Concordance index for discrimination measurement.
- Baseline hazard estimation for absolute risk prediction.

**Usage:**

```python
from ml.models.survival_coxph import CoxPHFitter

fitter = CoxPHFitter()
result = fitter.fit(rows, duration_col='duration_months',
                    event_col='target_defaulted',
                    feature_cols=['dti_ratio', 'interest_rate_pct'])
print(result.summary())

# "Each 10-point DTI increase raises hazard by 1.3x"
hr = result.hazard_ratios['dti_ratio']
print(f"DTI hazard ratio: {hr:.3f} per unit")
```

## Business Applications

| Use Case | Why Survival Matters |
| --- | --- |
| **IFRS 9 / CECL Provisioning** | Expected credit loss requires probability of default *at each time horizon*. Survival curves provide the term structure of PD. |
| **Risk-Based Pricing** | Borrowers with high early hazard should pay higher rates to compensate for near-term loss risk. |
| **Portfolio Stress Testing** | Shift the baseline hazard to simulate recession scenarios. |
| **Collection Prioritization** | Accounts with high hazard in the next 3 months should be flagged for early intervention. |
| **Champion/Challenger** | Compare survival curves across A/B arms to detect adverse late-term outcomes. |

## Integration with XGBoost

The survival models complement, not replace, the binary XGBoost classifier:

- **XGBoost** → accept/decline decision at application time (0.976 AUC-ROC).
- **Cox PH** → hazard ratios for regulatory explanation and sensitivity analysis.
- **Kaplan-Meier** → portfolio-level survival curves for board reporting.

The risk score from Cox PH can be combined with XGBoost's probability for a blended decision surface. See `ml/causal/` for A/B testing between survival-informed and binary-only arms.

## Time-Dependent AUC

Binary AUC is static — it measures lifetime discrimination. Time-dependent AUC measures how well the model separates defaulters from non-defaulters at each time horizon:

| Horizon | Typical AUC | Interpretation |
| --- | ---: | --- |
| 6 months | ~0.85 | Near-term risk (liquidity-driven defaults) |
| 12 months | ~0.90 | Medium-term risk (typical charge-off timing) |
| 24 months | ~0.88 | Longer-term risk (economic cycle exposure) |
| 36 months | ~0.85 | Full-term risk (includes paid-off loans as censored) |

## Dependencies

Add to `requirements-ml.txt`:

```
lifelines==0.30.*
scikit-survival==0.23.*
```

## References

- Cox, D. R. (1972). "Regression Models and Life-Tables." *Journal of the Royal Statistical Society.*
- Kaplan, E. L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations." *JASA.*
- Harrell, F. E. (2015). *Regression Modeling Strategies.* Springer.
