# A/B Testing Infrastructure — Champion/Challenger

## Overview

The A/B testing framework provides rigorous statistical comparison between a champion (heuristic) and challenger (ML) decision strategy. It builds on the deterministic assignment logic in `ml/governance/ab_test.py` and adds:

- Stratified randomization (by grade + purpose)
- Primary metric: profit lift with confidence intervals
- Secondary metrics: default rate, approval rate, confidence calibration
- Welch's t-test for profit difference
- Bootstrap 95% CIs for metric deltas
- Sequential testing support (early stopping boundaries)

## Configuration

A/B testing is controlled by two environment variables:

| Variable | Default | Description |
|---|---|---|
| `AB_TEST_ENABLED` | `false` | Set to `true`/`1`/`yes` to enable A/B routing |
| `AB_TEST_ML_RATIO` | `0.10` | Fraction of traffic routed to the ML (challenger) arm |

### Via the Experiment Framework

```python
from ml.causal.ab_framework import ExperimentConfig, ExperimentFramework

config = ExperimentConfig(
    name="phase_8_ml_vs_heuristic",
    control_arm="heuristic",
    treatment_arm="ml",
    alpha=0.05,
    min_sample_size=1000,
    stratified=True,
)
framework = ExperimentFramework(config)
```

## Stratified Randomization

When `stratified=True` (default), assignment is deterministic within strata defined by `grade + purpose`. The stratum key is hashed with SHA-256 and mapped to a 10,000-bucket space. Applications are split evenly (50/50) by default between control and treatment within each stratum.

This ensures:
- Balanced representation of risk grades across arms
- Balanced loan purposes across arms
- Deterministic replay: same application_id + grade + purpose always yields the same arm

## Metrics

### Primary: Profit Lift

Simulated profit per application:

```
profit = +0.12 × loan_amount  (performing approved loans)
profit = -0.65 × loan_amount  (defaulted approved loans)
profit =  0.00                (declined or manual review)
```

### Secondary

| Metric | Definition |
|---|---|
| `default_rate` | Defaults / Approved loans |
| `approval_rate` | Approved / Total applications |
| `calibration` | Mean decision confidence |

## Statistical Methodology

### Welch's t-test

Used for the profit lift comparison. Does not assume equal variance between arms. Computes:

```
t = (mean_treatment - mean_control) / sqrt(var_treatment/n_t + var_control/n_c)
```

Degrees of freedom use the Welch-Satterthwaite approximation. P-values are computed via `scipy.stats.t.sf` (with a pure-Python regularized incomplete beta fallback using Lentz's continued fraction method).

### Bootstrap Confidence Intervals

The 95% CI for profit delta is computed via the percentile bootstrap (10,000 resamples, seeded with `random.Random(42)` for reproducibility).

## Sample Size Requirements

- **Minimum per arm**: 1,000 applications (configurable via `min_sample_size`)
- Below this threshold, `compare_arms` returns `CONTINUE` regardless of observed effect
- No maximum — larger samples increase statistical power

## Interpreting Results

```python
from ml.causal.champion_challenger import compare_arms

result = framework.analyze(records)
decision = compare_arms(result)
```

| Recommendation | Condition | Action |
|---|---|---|
| **PROMOTE** | p < 0.05, CI entirely > 0, profit lift > 0 | Challenger outperforms champion |
| **ROLLBACK** | p < 0.05, CI entirely < 0, profit lift < 0 | Champion outperforms challenger |
| **CONTINUE** | Not significant, or insufficient sample | Keep both arms running |

### `ChampionChallengerDecision` Fields

| Field | Type | Description |
|---|---|---|
| `winner` | `str` | `"champion"`, `"challenger"`, or `"tie"` |
| `profit_lift` | `float` | Treatment profit minus control profit |
| `profit_ci` | `tuple[float, float]` | 95% confidence interval for profit lift |
| `p_value` | `float` | Two-sided p-value from Welch's t-test |
| `significant` | `bool` | True if p < alpha |
| `sample_size` | `int` | Total records across both arms |
| `recommendation` | `str` | Human-readable action message |

## Integration with Prometheus

When deployed, A/B metrics are exported via Prometheus counters:

| Metric | Labels | Description |
|---|---|---|
| `auditlend_ab_assignments_total` | `arm`, `grade`, `purpose` | Assignment count per arm per stratum |
| `auditlend_ab_outcomes_total` | `arm`, `decision` | Outcome count per arm |
| `auditlend_ab_profit_delta` | — | Current profit lift between arms |

These metrics are managed in `services/metrics.py` and should be updated after each call to `analyze()`.

## Running Tests

```bash
.venv/bin/python -m pytest tests/unit/test_ab_framework.py -q
```

## Example

```python
from ml.causal.ab_framework import ExperimentConfig, ExperimentFramework
from ml.causal.champion_challenger import compare_arms

config = ExperimentConfig(name="demo_experiment", min_sample_size=10)
framework = ExperimentFramework(config)

# Simulate some records
records = [
    {"arm": "heuristic", "decision": "APPROVE", "defaulted": 0,
     "loan_amount": 10000.0, "confidence": 0.85},
    {"arm": "ml", "decision": "APPROVE", "defaulted": 0,
     "loan_amount": 10000.0, "confidence": 0.92},
    # ... more records ...
]

result = framework.analyze(records)
decision = compare_arms(result)
print(decision.recommendation)  # e.g., "CONTINUE: insufficient sample..."
```

## Sequential Testing (Optional)

The framework supports early stopping boundaries for sequential monitoring. When configured, you can call `analyze` incrementally and apply stopping rules after each batch. This feature is opt-in and requires manual boundary specification.
