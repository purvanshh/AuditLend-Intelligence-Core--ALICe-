# AuditLend Intelligence Core (ALICe)

**Production credit risk ML system — 0.976 AUC-ROC, isotonic calibrated, SHAP explained, drift monitored.**

A calibrated XGBoost model trained on 1.3M Lending Club loans reduces default rates from 15.1% (heuristic) to 2.3% (ML), delivering a **$68.3M simulated profit delta** over the heuristic baseline on held-out 2018 data. Explanations are grounded in SHAP values, fairness is audited across proxy groups (max SPD 0.125), and every decision is preserved in an immutable audit trail for regulatory review.

---

## ML Performance at a Glance

| Metric | Raw Model | Calibrated (Isotonic) |
| --- | ---: | ---: |
| AUC-ROC | 0.9758 | **0.9757** |
| AUC-PR | 0.9367 | **0.9366** |
| Brier Score | 0.0266 | **0.0253** |
| ECE (Expected Calibration Error) | 0.0162 | **0.0036** |

**Calibration improvement:** Isotonic regression reduced ECE by **78%** (0.016 → 0.0036), ensuring predicted default probabilities are reliable across the entire risk spectrum.

## Business Impact

| Arm | Approval Rate | Default Rate (Approved) | Simulated Profit |
| --- | ---: | ---: | ---: |
| Heuristic (Rule-Based) | 85.14% | 15.06% | -$9.4M |
| XGB_V1 (ML) | 85.75% | **2.35%** | **+$58.9M** |
| **Delta** | **+0.6pp** | **-12.7pp** | **+$68.3M** |

On a held-out 2018 test set (49,230 loans), the ML scorer simultaneously **increased approvals** and **reduced defaults by 84%** relative to the heuristic baseline.

## What This System Does

- **Credit decision engine** with dual-path scoring: deterministic heuristic (RULE_SET_V1) and calibrated ML (RULE_SET_V2 with XGB_V1).
- **SHAP explainability** per prediction — every decision includes top-5 feature contributions.
- **Isotonic calibration** for reliable probability-of-default estimates across the full score range.
- **Proxy fairness audit** (SPD, EOD) across zip_code_prefix and employment_length_band groups.
- **Live drift detection** via Kolmogorov-Smirnov tests against a training-time reference snapshot.
- **Immutable audit trail** — all decisions, fallbacks, and data quality events are logged append-only.
- **A/B routing** between heuristic and ML arms for champion/challenger experimentation.
- **Deterministic mock providers** for credit bureau, bank analyzer, and GST verification.
- **Idempotent intake** with Redis + Postgres idempotency for safe retry and replay.

## Data & Training

| Detail | Value |
| --- | ---: |
| Source | Lending Club (2007–2018) |
| Training rows | 1,116,769 (2007–2016) |
| Validation rows | 156,290 (2017) |
| Test rows | 49,230 (2018) |
| Engineered features | 38 numeric + categorical |
| Algorithm | XGBoost (max_depth=6, lr=0.05, n_estimators=200) |
| Calibration | Isotonic regression (validation fit) |

## Fairness Reference Analysis

| Proxy Attribute | Reference Group | Max \|SPD\| | Max \|EOD\| |
| --- | --- | ---: | ---: |
| zip_code_prefix | 945 | 0.1247 | 0.0158 |
| employment_length_band | 10+ years | 0.0617 | 0.0086 |

Approval is the favorable outcome; equal opportunity is measured on the non-default class. Full breakdown in [model card](ml/models/reports/XGB_V1_evaluation.md).

## Architecture (Summary)

```
Client → FastAPI API → PostgreSQL + Redis → Celery Worker → Mock Providers
                          ↓
                   Audit Log (immutable)
                          ↓
              ML Scorer (XGB_V1 + SHAP + Drift)
```

Detailed architecture, pipeline walkthrough, and infrastructure layout: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## Business Impact Details

Profit model assumptions, sensitivity analysis, and risk reduction breakdown: **[docs/BUSINESS_IMPACT.md](docs/BUSINESS_IMPACT.md)**

## Quick Start

```bash
docker compose up --build -d
# or with ML scoring:
ML_ENABLED=true docker compose up --build -d
```

Submit a loan application:

```bash
APP_ID=$(curl -s -X POST http://localhost:8000/api/v1/apply-loan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-read-write" \
  -H "Idempotency-Key: smoke-001" \
  -d '{
    "idempotency_key": "smoke-001",
    "user_data": {
      "name": "Jane Doe", "pan": "ABCDE1234F",
      "monthly_income": 120000, "existing_emis": 25000,
      "loan_amount": 500000, "tenure_months": 36
    },
    "failure_flags": {
      "credit_bureau": "SUCCESS", "bank_analyzer": "SUCCESS", "gst_verifier": "SUCCESS"
    }
  }' | python3 -c 'import json,sys; print(json.load(sys.stdin)["application_id"])')
```

Check the decision:

```bash
curl -s -H "X-API-Key: dev-key-read-only" "http://localhost:8000/api/v1/decision/$APP_ID"
```

Full API reference: **[docs/ARCHITECTURE.md](#api-reference)** (in Architecture doc)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `api/` | FastAPI routes, auth, schemas |
| `worker/` | Celery tasks, outbox poller |
| `engine/` | Scoring, rules, decision orchestration, explanations |
| `ml/` | Model training, calibration, SHAP explainability, governance, benchmark |
| `ml/models/reports/` | Evaluation reports, model cards, fairness audits |
| `services/` | Provider clients, crypto, audit safety, metrics |
| `models/` | SQLAlchemy ORM models |
| `mock_apis/` | Deterministic external-provider mocks |
| `tests/` | Unit, integration, and chaos tests |
| `docs/` | Architecture, calibration, business impact, security |
| `research/` | EDA, WOE/IV, survival analysis, causal inference |

## Testing

```bash
.venv/bin/pytest tests/unit -q              # unit tests
.venv/bin/pytest tests -q --cov=engine --cov=ml --cov=services --cov=worker --cov-report=term  # full suite
```

Full ML evaluation:

```bash
LENDING_CLUB_DATA_PATH="ml/data/raw/accepted_2007_to_2018Q4.csv.gz" \
  .venv/bin/python -m ml.models.evaluate --official-xgb-v1
```

## Project Principles

- **Determinism** — identical inputs produce identical decisions. No randomness in scoring.
- **Idempotency** — safe retry and replay without duplicate decisions.
- **Auditability** — every decision step is logged in an immutable append-only audit trail.
- **Calibration** — predicted probabilities match observed frequencies (ECE 0.0036).
- **Explainability** — SHAP values power per-prediction factor contributions.
- **Fairness monitoring** — proxy group diagnostics and drift detection at inference time.
- **Separation of concerns** — risk score, data reliability, and confidence are distinct.
- **Defense in depth** — fallback data reduces confidence, manual review catches edge cases.

---

*Full architecture, configuration, and operational details: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.*
