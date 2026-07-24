# AuditLend Intelligence Core (ALICe)

**Audit-grade credit decision engine — ML scoring, SHAP explainability, drift monitoring, portfolio analytics, production hardened.**

A calibrated XGBoost model (0.976 AUC-ROC, 0.0036 ECE) trained on 1.3M Lending Club loans reduces default rates from 15.1% (heuristic) to 2.3% (ML), delivering a **$68.3M simulated profit delta**. The system encompasses the full ML lifecycle: experiment tracking (MLflow), model registry, drift detection (Evidently + KS-test), ONNX export, prediction caching, A/B experimentation, causal inference, uplift modeling, portfolio risk aggregation, and an immutable compliance audit trail.

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
- **SHAP explainability** per prediction — every decision includes top-8 feature contributions.
- **LLM narrative explanations** — natural-language decision summaries via optional LLM integration.
- **Policy RAG** — regulatory policy grounding with ChromaDB vector search (optional).
- **Multi-modal document parsing** — Tesseract-based OCR for scanned financial documents (PDF, images).
- **Isotonic calibration** for reliable probability-of-default estimates across the full score range.
- **Proxy fairness audit** (SPD, EOD) across zip_code_prefix and employment_length_band groups.
- **Live drift detection** via Kolmogorov-Smirnov tests (with Evidently-powered dashboards as optional upgrade).
- **Model registry & experiment tracking** — file-backed ModelRegistry with optional MLflow integration.
- **ONNX export & prediction cache** — optimized inference paths with LRU+TTL caching.
- **Batch prediction API** — async batch scoring with configurable batch sizes.
- **A/B experimentation framework** — randomized arm assignment for heuristic vs ML champion/challenger trials.
- **Causal inference** — propensity score matching (PSM) and synthetic control for treatment effect estimation.
- **Uplift modeling** — individual treatment effect (ITE) prediction with UpliftXGB.
- **Survival analysis** — Kaplan-Meier and CoxPH models for time-to-default estimation.
- **Portfolio risk analytics** — concentration analysis (HHI), stress testing, portfolio summary reports.
- **CLI tool** — `auditlend-portfolio` for portfolio analysis from the command line.
- **Immutable audit trail** — all decisions, fallbacks, and data quality events are logged append-only.
- **Deterministic mock providers** for credit bureau, bank analyzer, and GST verification.
- **Idempotent intake** with Redis + Postgres idempotency for safe retry and replay.
- **OAuth2/OIDC auth** — CompositeAuth supports both APIKey and JWT bearer tokens.
- **Rate limiting** — per-client in-memory token bucket with environment-configurable limits.
- **Security headers** — HSTS, CSP, X-Frame-Options, and friends applied by middleware.
- **Vault integration** — optional HashiCorp Vault for PII encryption key management.

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
┌────────────┐   ┌───────────────────────────────────────────────────────────────┐
│  Clients   │──▶│                    FastAPI API (port 8000)                     │
│ (REST/CLI) │   │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌─────────────┐  │
└────────────┘   │  │ API Key  │  │  OAuth2  │  │  Rate     │  │  Security   │  │
                 │  │  Auth    │  │ /OIDC    │  │  Limiter  │  │  Headers    │  │
                 │  └──────────┘  └──────────┘  └───────────┘  └─────────────┘  │
                 │  Routes: /apply-loan /decision /explain /batch /monitoring   │
                 └────────────────────────┬──────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │    PostgreSQL    │  │      Redis       │  │    Celery        │
         │  - Applications  │  │  - Idempotency   │  │    Worker        │
         │  - Audit Log     │  │  - Cache         │  │  - Process app   │
         │  - External Data │  │  - Circuit       │  │  - Score ML      │
         │  - Decisions     │  │    Breaker       │  │  - Call mocks    │
         └──────────────────┘  └──────────────────┘  └────────┬─────────┘
                                                              │
                     ┌────────────────────────────────────────┼──────────────┐
                     ▼                 ▼                      ▼              ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │   Credit     │  │    Bank      │  │     GST      │  │   MLflow     │
          │   Bureau     │  │   Analyzer   │  │   Verifier   │  │  (optional)  │
          │   (mock)     │  │   (mock)     │  │   (mock)     │  │  Tracking    │
          └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                               ML Layer (optional)                                │
│                                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ XGB_V1   │  │  SHAP    │  │  ONNX    │  │  Drift   │  │  Experiment      │  │
│  │  Model   │  │ Explain  │  │  Export   │  │  Detect  │  │  Tracker (MLflow)│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Uplift  │  │  Causal  │  │ Survival │  │ Portfolio│  │  Prediction      │  │
│  │  XGB     │  │  PSM/SC  │  │  KM/Cox  │  │  Risk    │  │  Cache           │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                                                  │
│  Monitoring: Prometheus + Grafana + Evidently (drift dashboards)                  │
│  Governance: Model Registry + A/B Framework + Policy RAG                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Detailed architecture, pipeline walkthrough, and infrastructure layout: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## Business Impact Details

Profit model assumptions, sensitivity analysis, and risk reduction breakdown: **[docs/BUSINESS_IMPACT.md](docs/BUSINESS_IMPACT.md)**

## Quick Start

```bash
docker compose up --build -d
# or with ML scoring:
ML_ENABLED=true docker compose up --build -d
# or with MLflow experiment tracking:
ML_ENABLED=true MLFLOW_ENABLED=true docker compose up --build -d
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

Batch prediction (requires ML enabled):

```bash
curl -s -X POST http://localhost:8000/api/v1/batch/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-read-write" \
  -d '{
    "features": [{"monthly_income": 120000, "loan_amount": 500000, "credit_score_midpoint": 700}],
    "model_version": "XGB_V1"
  }'
```

CLI portfolio analysis:

```bash
auditlend-portfolio summary --input decisions.json --output report.md
auditlend-portfolio stress-test --input decisions.json --scenario recession --shock 0.15
auditlend-portfolio html-report --input decisions.json --output report.html
```

Full API reference: **[docs/ARCHITECTURE.md](#api-reference)** (in Architecture doc)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `api/` | FastAPI routes, auth (APIKey + OAuth2/OIDC), schemas, rate limiting |
| `worker/` | Celery tasks, outbox poller |
| `engine/` | Scoring, rules, decision orchestration, explanations |
| `ml/models/` | Training pipeline (XGBoost), evaluation, calibration, survival models, uplift XGB |
| `ml/explain/` | SHAP explainer, LLM narrative generator, policy RAG |
| `ml/governance/` | Model registry, drift detection, A/B framework |
| `ml/optimize/` | ONNX export, prediction cache, async model loader |
| `ml/causal/` | A/B framework, propensity score matching, synthetic control |
| `ml/portfolio/` | Portfolio risk aggregation, stress testing, CLI tooling |
| `ml/monitoring/` | Evidently-powered drift reporting & dashboards |
| `services/` | Provider clients, crypto, Vault integration, audit, metrics |
| `models/` | SQLAlchemy ORM models |
| `mock_apis/` | Deterministic external-provider mocks |
| `cli/` | `auditlend-portfolio` CLI entry point |
| `demo/` | Portfolio analysis demo script |
| `grafana/` | Grafana dashboards (drift monitoring) |
| `tests/` | Unit, integration, and chaos tests |
| `docs/` | Architecture, calibration, business impact, survival analysis, A/B testing, policy corpus |
| `research/` | EDA, WOE/IV, survival analysis, causal inference notebooks |

## Testing

```bash
.venv/bin/pytest tests/unit -q                          # all unit tests (437+ tests)
.venv/bin/pytest tests/unit/test_scoring.py -q          # engine tests
.venv/bin/pytest tests/unit/test_experiment_tracking.py -q  # MLflow tracking tests
.venv/bin/pytest tests/unit/test_drift_reporter.py -q   # drift detection tests
.venv/bin/pytest tests/unit/test_portfolio_risk.py -q   # portfolio analytics tests
```

Full ML evaluation:

```bash
LENDING_CLUB_DATA_PATH="ml/data/raw/accepted_2007_to_2018Q4.csv.gz" \
  .venv/bin/python -m ml.models.evaluate --official-xgb-v1
```

Portfolio CLI demo (no Docker needed):

```bash
.venv/bin/python -m cli.portfolio summary --input demo/sample_decisions.json --output /tmp/report.md
```

## Optional Dependencies

Most ML and security features degrade gracefully when their dependencies are missing:

| Feature | Dependency | Env Guard |
| --- | --- | --- |
| MLflow experiment tracking | `mlflow` | `MLFLOW_ENABLED=true` |
| Evidently drift dashboards | `evidently` | auto-detected |
| ONNX model export | `onnx`, `onnxmltools`, `onnxruntime` | auto-detected |
| OAuth2/OIDC auth | `python-jose[cryptography]` | `OIDC_ISSUER` config |
| Vault secrets management | `hvac` | `VAULT_URL` config |
| LLM narrative explanations | `litellm` / `openai` | auto-detected |
| Policy RAG | `chromadb` | auto-detected |
| Document OCR parsing | `pytesseract`, `pdf2image`, `Pillow` | auto-detected |

## Project Principles

- **Determinism** — identical inputs produce identical decisions. No randomness in scoring.
- **Idempotency** — safe retry and replay without duplicate decisions.
- **Auditability** — every decision step is logged in an immutable append-only audit trail.
- **Calibration** — predicted probabilities match observed frequencies (ECE 0.0036).
- **Explainability** — SHAP, LLM narrative, and policy-grounded explanations for every decision.
- **Fairness monitoring** — proxy group diagnostics and drift detection at inference time.
- **Graceful degradation** — all heavy dependencies (MLflow, Evidently, ONNX, Vault, OIDC) are optional; the system works without them.
- **Separation of concerns** — risk score, data reliability, and confidence are distinct.
- **Defense in depth** — fallback data reduces confidence, manual review catches edge cases.
- **Production hardening** — rate limiting, security headers, request size limits, and composable auth (APIKey + OAuth2/OIDC).

---

*Full architecture, configuration, and operational details: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.*
