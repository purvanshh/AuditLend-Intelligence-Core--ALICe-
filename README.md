# AuditLend Intelligence Core (ALICe)

AuditLend Intelligence Core (ALICe) is an auditable credit decision engine with deterministic failure simulation, idempotent intake, asynchronous processing, encrypted PII storage, immutable audit logs, calibrated decision confidence, human-readable explanations, and an opt-in ML scorer with fallback guardrails.

This repository is a production-readiness reference implementation built around local infrastructure, deterministic mock providers, Docker Compose, PostgreSQL, Redis, FastAPI, Celery, and a governed ML stack trained on Lending Club historical data. It is not wired to real credit bureaus or live repayment ledgers, but the current repository state is validated with a full held-out `XGB_V1` evaluation run, zero-skip automated tests, live Docker ML smoke tests, proxy fairness diagnostics, and drift-alert coverage.

## Current Status

Last verified locally: **2026-05-03**

| Area | Status | Evidence |
| --- | --- | --- |
| Full repository test suite | PASS | `./.venv/bin/pytest tests -q --cov=api --cov=engine --cov=ml --cov=services --cov=worker --cov-report=term` -> `187 passed` |
| Unit suite | PASS | `./.venv/bin/pytest tests/unit -q` -> included in the full pass above |
| Integration + chaos slice | PASS | covered in the zero-skip full suite above |
| Coverage report | PASS | `./.venv/bin/pytest tests -q --cov=api --cov=engine --cov=ml --cov=services --cov=worker --cov-report=term` -> `86%` |
| Official XGB_V1 benchmark | PASS | `python -m ml.benchmark.heuristic_vs_ml --official-xgb-v1 --ml-threshold 0.5` |
| Docker ML E2E smoke | PASS | `ML_ENABLED=true docker compose up --build -d` plus live `/apply-loan`, `/decision`, and `/explanation` verification on May 3, 2026 |

The current official ML benchmark from the May 3, 2026 verification run:

```json
{
  "heuristic": {
    "approval_rate": 0.851371,
    "default_rate_on_approved": 0.15055,
    "simulated_profit": -9354600.5
  },
  "ml": {
    "approval_rate": 0.857526,
    "default_rate_on_approved": 0.023498,
    "simulated_profit": 58939506.5
  },
  "profit_delta_ml_minus_heuristic": 68294107.0
}
```

## What Is Actually Implemented

### API and Processing

- FastAPI application with authenticated loan application intake.
- Celery worker for asynchronous processing.
- Redis broker/result backend, idempotency cache, and circuit breaker state.
- PostgreSQL source of truth for applications, idempotency records, external data snapshots, outbox messages, and audit logs.
- Transactional outbox pattern so API writes and task delivery intent are committed together.
- Worker-side task claiming with atomic `UPDATE ... WHERE status=PENDING`.
- External data fetch reuse so worker retries do not repeat already-persisted provider calls.

### Decision Engine

- Weighted risk score in `engine/scoring.py`.
- Immutable rule set registry in `engine/rule_sets.py`.
- Decision rules in `engine/rules.py`.
- Orchestration in `engine/decision.py`.
- Separate fields for:
  - `risk_score`
  - `data_reliability`
  - calibrated `confidence`
- GST non-compliance gate that prevents automatic approval when GST mismatch is explicit.
- Manual review override when calibrated confidence falls below threshold.
- `RULE_SET_V2` path for ML-backed decisions with heuristic shadow scoring and deterministic fallback.

### ML Platform

- Lending Club ingestion, feature engineering, and time-based splitting under `ml/data/`.
- Model training, evaluation, and isotonic calibration under `ml/models/`.
- SHAP-based per-prediction explainability under `ml/explain/`.
- File-backed model registry and KS-based drift detection under `ml/governance/`.
- Deterministic A/B routing plus heuristic-versus-ML benchmark utilities.
- `ML_SCORING` audit entries with `model_version`, `scoring_strategy`, fallback reason, and feature contributions.

### Security and Data Protection

- API key authentication on application, status, decision, and explanation routes.
- AES-256-GCM encryption for stored application PII.
- Salted SHA-256 PAN hash.
- Raw PAN is not stored in loan application rows.
- Audit snapshots are sanitized and band raw financial values before persistence.
- Explanation responses are derived from audit logs and do not expose raw PAN or names.

### Audit and Compliance Trail

- Append-only audit log writes in application code.
- Database trigger blocks `UPDATE` and `DELETE` against `audit_logs`.
- Audit entries include step, input/output snapshot, error type, fallback flag, rule version, actor, and timestamp where applicable.
- Explanation engine builds summaries and timelines from audit history, not from a fresh recomputation.

### Resilience and Observability

- Retry/backoff for retryable provider failures.
- Redis-backed circuit breaker with half-open single-probe lock.
- Per-provider failure modes.
- Configurable external API timeout.
- Worker health endpoint on port `8004`.
- Structured JSON logs.
- Prometheus metrics for applications, external calls, circuit breaker state, decision confidence, task duration, task failures, drift alerts, and A/B experiment arms.

### Deterministic Mocks

Mock services live under `mock_apis/` and support deterministic success and failure modes:

- Credit bureau: `SUCCESS`, `TIMEOUT`, `STALE_DATA`, `SERVICE_DOWN`
- Bank analyzer: `SUCCESS`, `PARTIAL_DATA`, `FORMAT_ERROR`
- GST verifier: `SUCCESS`, `PAN_MISMATCH`, `NO_RECORD`

For identical inputs, mock responses are deterministic. Request IDs are input-derived, not random UUIDs. Stale data uses a fixed reference date.

## What This Project Does Not Claim

This is important. AuditLend is built end to end, but it is not a drop-in live lending system without additional institutional controls.

- The heuristic scorecard remains deterministic and governed, but it is not itself empirically calibrated against a real repayment/default dataset.
- The ML stack is implemented end to end, and the published metrics in this README come from the official held-out `XGB_V1` evaluation, benchmark, Docker smoke validation, and repository test suite captured on May 3, 2026.
- Mock APIs are deterministic test doubles, not real provider integrations.
- API key auth is suitable for this reference stack, but a real deployment should use OAuth2/OIDC, scoped service identities, mTLS, and centralized secret management.
- Docker Compose is a local/demo deployment target, not a production orchestrator.
- TLS/mTLS, managed key rotation, SIEM integration, data retention automation, and formal model risk governance are outside this repository.
- Containers currently run in a local developer-oriented configuration; hardening user privileges and image policies is a production deployment task.

## Architecture

```text
Client
  |
  | POST /api/v1/apply-loan
  v
FastAPI API
  | \
  |  \-- PostgreSQL
  |       - loan_applications
  |       - idempotency_records
  |       - outbox
  |       - external_data
  |       - audit_logs
  |
  \-- Redis
       - idempotency cache
       - Celery broker/result backend
       - circuit breaker state

Celery Worker
  |
  |-- polls outbox
  |-- atomically claims applications
  |-- fetches/reuses external data
  |-- computes decision
  |-- stores audit trail
  |
  |-- Credit Bureau Mock
  |-- Bank Analyzer Mock
  \-- GST Verifier Mock
```

## System Pipeline

1. `POST /api/v1/apply-loan` validates input, checks idempotency, encrypts PII, writes the application plus outbox intent, and returns immediately.
2. The worker polls the outbox, atomically claims the application, and reuses any already-persisted external snapshots on retry.
3. Credit bureau, bank analyzer, and GST verifier mocks are called deterministically with typed failure handling and audit-safe persistence.
4. The decision engine computes heuristic risk, data reliability, calibrated confidence, and, when enabled, invokes `XGB_V1` with guardrails, SHAP explanations, and live drift checks.
5. ML scoring either stays active under `RULE_SET_V2` or falls back deterministically to `RULE_SET_V1` when confidence, artifact availability, or forced failure flags require it.
6. Audit logs record every step, including `ML_SCORING`, `DRIFT_DETECTED`, fallbacks, and the final decision snapshot used by `/explanation`.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `api/` | FastAPI app, auth, routes, schemas |
| `worker/` | Celery app, outbox poller, processing task |
| `engine/` | Scoring, rule sets, decision orchestration, explanation builder |
| `services/` | Provider clients, crypto, audit safety, metrics, logging |
| `models/` | SQLAlchemy models |
| `migrations/` | Alembic migrations |
| `mock_apis/` | Deterministic external-provider mocks |
| `tests/` | Unit, integration, and chaos tests |
| `docs/CALIBRATION.md` | Heuristic and ML calibration notes, smoke metrics, and rule-set governance |
| `ml/` | Data, models, explainability, governance, benchmark scripts, and developer docs |

## ML Data Setup

The Lending Club training dataset lives under `ml/data/raw/` and is intentionally excluded from git and Docker image layers.

Set the dataset path with:

```bash
export LENDING_CLUB_DATA_PATH="ml/data/raw/accepted_2007_to_2018Q4.csv.gz"
```

If your local Kaggle export is still unpacked into nested folders, point `LENDING_CLUB_DATA_PATH` at the real CSV path instead. See `ml/data/README.md` for the canonical path and download notes.

For the full ML workflow, including training, calibration, explanation, registry, drift, and benchmark commands, see `ml/README.md`.

## ML Model Performance

The repository now includes an official signed-off artifact set under `ml/models/`:

- `ml/models/XGB_V1_model.pkl`
- `ml/models/XGB_V1_calibrator.pkl`
- `ml/models/XGB_V1_features.json`
- `ml/models/manifest.yaml`

Those artifacts were trained on the full modeled Lending Club corpus with the PRD-aligned split:

| Split | Rows |
| --- | ---: |
| Train (2007-2016) | 1,116,769 |
| Validation (2017) | 156,290 |
| Test (2018) | 49,230 |

Held-out evaluation metrics from `ml/models/reports/XGB_V1_evaluation.md`:

| Metric | Raw Test | Calibrated Test |
| --- | ---: | ---: |
| AUC-ROC | `0.975786` | `0.975664` |
| AUC-PR | `0.936718` | `0.936609` |
| Brier score | `0.026582` | `0.025293` |
| ECE | `0.016177` | `0.003550` |

Benchmark metrics from `ml/benchmark/reports/XGB_V1_heuristic_vs_ml.md` at the `0.50` calibrated default-probability threshold:

| Arm | Approval Rate | Default Rate on Approved | Simulated Profit |
| --- | ---: | ---: | ---: |
| Heuristic | `0.851371` | `0.150550` | `-9354600.50` |
| XGB_V1 | `0.857526` | `0.023498` | `58939506.50` |

On this held-out 2018 benchmark, `XGB_V1` improved both headline PRD directions at the chosen threshold: approval rate increased by `0.006155` while default rate on approved loans dropped by `0.127052`.

Proxy fairness analysis from `ml/models/reports/XGB_V1_evaluation.md` on the held-out 2018 test split:

| Proxy Attribute | Reference Group | Max \|SPD\| | Max \|EOD\| | Largest Observed Gap |
| --- | --- | ---: | ---: | --- |
| `zip_code_prefix` | `945` | `0.124725` | `0.015766` | prefix `104` had approval-rate SPD `-0.124725` vs the reference group |
| `employment_length_band` | `10+` | `0.061652` | `0.008574` | band `0` had approval-rate SPD `-0.061652` vs the reference group |

These are reference fairness diagnostics based on proxy groupings from Lending Club data, not protected-class measurements. Approval is treated as the favorable outcome, and equal opportunity is measured on the non-default class.

Install ML-only dependencies separately from the core API/runtime stack:

```bash
pip install -r requirements-ml.txt
```

## Quick Start

Prerequisites:

- Docker Desktop or Docker Engine
- Docker Compose

This project requires real-looking local secrets. For local development, `docker-compose.override.yml` supplies dev-only values and is gitignored.

Start the default stack:

```bash
docker compose up --build -d
```

Start the ML-enabled stack:

```bash
ML_ENABLED=true docker compose up --build -d
```

Check service health:

```bash
curl http://localhost:8000/health
curl http://localhost:8004/health
docker compose ps
```

Expected API health response:

```json
{"status":"healthy","service":"auditlend-api","version":"2.0.0"}
```

Useful local endpoints:

| Service | URL |
| --- | --- |
| API | `http://localhost:8000` |
| Credit bureau mock | `http://localhost:8001` |
| Bank analyzer mock | `http://localhost:8002` |
| GST verifier mock | `http://localhost:8003` |
| Worker health | `http://localhost:8004/health` |
| Flower | `http://localhost:5555` |
| Metrics | `http://localhost:8000/metrics` |

## API Authentication

Protected routes require `X-API-Key`.

The local override defines:

```text
dev-key-read-write
dev-key-read-only
```

Examples:

```bash
curl -i http://localhost:8000/api/v1/status/00000000-0000-0000-0000-000000000000
```

Expected: `401`

```bash
curl -i \
  -H "X-API-Key: dev-key-read-only" \
  http://localhost:8000/api/v1/status/00000000-0000-0000-0000-000000000000
```

Expected: `404` because authentication passed and the application does not exist.

## End-to-End Smoke Test

Submit an application:

```bash
APP_ID=$(curl -s -X POST http://localhost:8000/api/v1/apply-loan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-read-write" \
  -H "Idempotency-Key: smoke-001" \
  -d '{
    "idempotency_key": "smoke-001",
    "user_data": {
      "name": "Jane Doe",
      "pan": "ABCDE1234F",
      "monthly_income": 120000,
      "existing_emis": 25000,
      "loan_amount": 500000,
      "tenure_months": 36
    },
    "failure_flags": {
      "credit_bureau": "SUCCESS",
      "bank_analyzer": "SUCCESS",
      "gst_verifier": "SUCCESS"
    }
  }' | python3 -c 'import json,sys; print(json.load(sys.stdin)["application_id"])')

echo "$APP_ID"
```

Check status:

```bash
curl -s \
  -H "X-API-Key: dev-key-read-only" \
  "http://localhost:8000/api/v1/status/$APP_ID"
```

Expected terminal shape:

```json
{
  "application_id": "uuid",
  "status": "COMPLETED",
  "updated_at": "timestamp"
}
```

Check decision:

```bash
curl -s \
  -H "X-API-Key: dev-key-read-only" \
  "http://localhost:8000/api/v1/decision/$APP_ID"
```

Expected shape:

```json
{
  "application_id": "uuid",
  "decision": "APPROVE",
  "confidence": 0.7,
  "data_reliability": 1.0,
  "risk_score": 64.1,
  "rule_version": "RULE_SET_V1",
  "model_version": null,
  "scoring_strategy": "heuristic",
  "ab_test_arm": null
}
```

Check explanation:

```bash
curl -s \
  -H "X-API-Key: dev-key-read-only" \
  "http://localhost:8000/api/v1/explanation/$APP_ID"
```

Expected shape:

```json
{
  "application_id": "uuid",
  "decision": "APPROVE",
  "summary": "Decision APPROVE was produced from verified data sources with confidence 0.70.",
  "timeline": [
    {"step": "PROCESSING_STARTED", "status": "PROCESSING"},
    {"step": "CREDIT_BUREAU_FETCH", "status": "SUCCESS"},
    {"step": "GST_VERIFIER_FETCH", "status": "SUCCESS"},
    {"step": "BANK_ANALYZER_FETCH", "status": "SUCCESS"},
    {"step": "DECISION_CALCULATION", "status": "APPROVE"}
  ],
  "rule_version": "RULE_SET_V1"
}
```

Replay the same request with the same idempotency key and payload. Expected: `200` and the same `application_id`.

Reuse the same idempotency key with a different payload. Expected: `409 Conflict`.

### Verified ML-Enabled Success Path

The following local verification run was executed on **May 3, 2026** with `ML_ENABLED=true` and the official `XGB_V1` artifacts mounted into the worker.

Input profile:

```json
{
  "idempotency_key": "phase4-ml-success-002",
  "user_data": {
    "name": "Phase Four Success",
    "pan": "AAAAA1234F",
    "monthly_income": 200000,
    "existing_emis": 5000,
    "loan_amount": 50000,
    "tenure_months": 36,
    "purpose": "credit_card",
    "home_ownership": "MORTGAGE"
  },
  "failure_flags": {
    "credit_bureau": "SUCCESS",
    "bank_analyzer": "SUCCESS",
    "gst_verifier": "SUCCESS"
  }
}
```

Observed decision excerpt:

```json
{
  "application_id": "55063108-c22a-4c44-81d0-dd38bbec5e81",
  "decision": "APPROVE",
  "confidence": 1.0,
  "data_reliability": 1.0,
  "risk_score": 96.21,
  "rule_version": "RULE_SET_V2",
  "model_version": "XGB_V1",
  "scoring_strategy": "ml",
  "ab_test_arm": null
}
```

Observed explanation excerpt:

```json
{
  "decision": "APPROVE",
  "model_version": "XGB_V1",
  "summary": "Decision APPROVE was produced from verified data sources with confidence 1.00. Model factors: Employment Length Years (0) increased predicted default risk and Interest Rate Pct (0) increased predicted default risk, while Credit Score Recent Delta (0) reduced predicted default risk.",
  "model_factor_contributions": [
    {
      "feature_name": "Credit Score Recent Delta",
      "raw_value": "0",
      "shap_contribution": -1.70156,
      "direction": "decrease_default_risk"
    }
  ]
}
```

### Verified Forced Fallback Path

The same high-confidence profile was re-run with `failure_flags.ml_model = "FORCE_LOW_CONFIDENCE"` to prove the deterministic guardrail fallback branch.

Observed decision excerpt:

```json
{
  "application_id": "f2c6f216-2a1a-4379-a4f0-59c733955e93",
  "decision": "APPROVE",
  "confidence": 1.0,
  "data_reliability": 1.0,
  "risk_score": 83.62,
  "rule_version": "RULE_SET_V1",
  "model_version": "XGB_V1",
  "scoring_strategy": "heuristic",
  "ab_test_arm": null
}
```

Observed explanation excerpt:

```json
{
  "decision": "APPROVE",
  "model_version": "XGB_V1",
  "summary": "Decision APPROVE was produced with degraded data quality. Data quality issues recorded: Ml Scoring: fallback. Confidence is 1.00. ML confidence was forced low for testing, so the heuristic scorer was used instead.",
  "model_factor_contributions": []
}
```

## Deterministic Failure Scenarios

All examples require `X-API-Key: dev-key-read-write` for submit and `X-API-Key: dev-key-read-only` for reads.

### Credit Bureau Timeout

Input:

```json
"failure_flags": {
  "credit_bureau": "TIMEOUT",
  "bank_analyzer": "SUCCESS",
  "gst_verifier": "SUCCESS"
}
```

Expected behavior:

- Credit provider retries and falls back to conservative credit score `600`.
- Data reliability is reduced.
- Calibrated confidence can force `NEEDS_REVIEW`.
- Audit log includes `CREDIT_BUREAU_FETCH` with `TIMEOUT` and fallback information.

### Partial Bank Data

Input:

```json
"failure_flags": {
  "credit_bureau": "SUCCESS",
  "bank_analyzer": "PARTIAL_DATA",
  "gst_verifier": "SUCCESS"
}
```

Expected behavior:

- Missing income stability is filled with neutral `0.5`.
- Data reliability is reduced.
- Decision may still approve if the risk score and confidence remain sufficient.
- Explanation marks the degraded bank data path.

### Total External Data Meltdown

Input:

```json
"failure_flags": {
  "credit_bureau": "SERVICE_DOWN",
  "bank_analyzer": "FORMAT_ERROR",
  "gst_verifier": "NO_RECORD"
}
```

Expected behavior:

- Conservative fallbacks are applied.
- Data reliability drops sharply.
- Application routes to manual review.
- Audit timeline records each failing source.

### GST PAN Mismatch

Input:

```json
"failure_flags": {
  "credit_bureau": "SUCCESS",
  "bank_analyzer": "SUCCESS",
  "gst_verifier": "PAN_MISMATCH"
}
```

Expected behavior:

- GST compliance is explicitly false.
- Effective risk score is capped below automatic approval.
- Application routes to review unless another decline rule applies.

## API Reference

### `POST /api/v1/apply-loan`

Creates a loan application and records an outbox message for asynchronous processing.

Headers:

```text
Content-Type: application/json
X-API-Key: dev-key-read-write
Idempotency-Key: unique-logical-request-key
```

Request:

```json
{
  "idempotency_key": "req-001",
  "user_data": {
    "name": "Jane Doe",
    "pan": "ABCDE1234F",
    "monthly_income": 120000,
    "existing_emis": 25000,
    "loan_amount": 500000,
    "tenure_months": 36,
    "bank_statement": []
  },
  "failure_flags": {
    "credit_bureau": "SUCCESS",
    "bank_analyzer": "SUCCESS",
    "gst_verifier": "SUCCESS"
  }
}
```

Responses:

| Code | Meaning |
| --- | --- |
| `201` | New application accepted |
| `200` | Same idempotency key and same payload replayed |
| `400` | Validation error |
| `401` | Missing or invalid API key |
| `403` | API key lacks required scope |
| `409` | Same idempotency key reused with a different payload |

### `GET /api/v1/status/{application_id}`

Requires read scope.

```json
{
  "application_id": "uuid",
  "status": "PENDING|PROCESSING|COMPLETED|MANUAL_REVIEW",
  "updated_at": "timestamp"
}
```

### `GET /api/v1/decision/{application_id}`

Requires read scope.

If complete:

```json
{
  "application_id": "uuid",
  "decision": "APPROVE|DECLINE|NEEDS_REVIEW",
  "confidence": 0.7,
  "data_reliability": 1.0,
  "risk_score": 64.1,
  "factors": ["risk_score (computed) = 64.10"],
  "rule_version": "RULE_SET_V1"
}
```

If still processing, the endpoint returns `202`.

### `GET /api/v1/explanation/{application_id}`

Requires read scope.

```json
{
  "application_id": "uuid",
  "decision": "APPROVE",
  "summary": "Human-readable explanation",
  "factors": [
    {"name": "Risk Score", "value": "64.10", "status": "computed"}
  ],
  "model_factor_contributions": [],
  "timeline": [
    {"step": "DECISION_CALCULATION", "status": "APPROVE", "timestamp": "timestamp"}
  ],
  "rule_version": "RULE_SET_V1",
  "model_version": null,
  "generated_at": "timestamp"
}
```

### `GET /metrics`

No API key required in this local stack.

Prometheus series include:

- `auditlend_applications_total`
- `auditlend_external_api_requests_total`
- `auditlend_external_api_latency_seconds`
- `auditlend_circuit_breaker_state`
- `auditlend_decision_confidence`
- `auditlend_task_duration_seconds`
- `auditlend_task_failures_total`
- `auditlend_drift_alerts_total`
- `auditlend_ab_assignments_total`
- `auditlend_ab_decisions_total`
- `auditlend_ab_decision_confidence`

### Error Format

Errors use Problem Details style:

```json
{
  "type": "https://api.auditlend.local/errors/validation",
  "title": "Validation Error",
  "detail": "monthly_income must be positive",
  "instance": "/api/v1/apply-loan"
}
```

## Configuration

Production secrets are not committed. Local dev-only values can live in `docker-compose.override.yml`, which is ignored by git.

| Variable | Purpose | Compose value |
| --- | --- | --- |
| `DATABASE_URL` | Sync SQLAlchemy/Postgres URL | `postgresql://auditlend:auditlend@postgres:5432/auditlend` |
| `ASYNC_DATABASE_URL` | Async SQLAlchemy/Postgres URL | `postgresql+asyncpg://auditlend:auditlend@postgres:5432/auditlend` |
| `AUDITLEND_ASYNC_DB_POOL` | Optional async DB pool mode; `null` is used by multi-event-loop tests only | `pooled` |
| `REDIS_URL` | Celery, idempotency cache, circuit breaker state | `redis://redis:6379/0` |
| `IDEMPOTENCY_CACHE_TTL_SECONDS` | Redis idempotency replay TTL | `86400` |
| `CREDIT_BUREAU_URL` | Credit mock base URL | `http://credit-bureau:8001` |
| `BANK_ANALYZER_URL` | Bank mock base URL | `http://bank-analyzer:8002` |
| `GST_VERIFIER_URL` | GST mock base URL | `http://gst-verifier:8003` |
| `CONFIDENCE_THRESHOLD` | Manual review confidence threshold | `0.6` |
| `PII_ENCRYPTION_KEY` | Required 64-character hex AES-256-GCM key | `${PII_ENCRYPTION_KEY}` |
| `PAN_HASH_SALT` | Required per-environment PAN hash salt | `${PAN_HASH_SALT}` |
| `API_KEYS` | Comma-separated API keys with optional scopes | `${API_KEYS}` |
| `CORS_ALLOWED_ORIGINS` | Comma-separated trusted browser origins; wildcard rejected | `http://localhost:3000,http://localhost:8000` |
| `CIRCUIT_BREAKER_THRESHOLD` | Failures before opening circuit | `5` |
| `CIRCUIT_BREAKER_WINDOW_SECONDS` | Circuit failure counting window | `60` |
| `CIRCUIT_BREAKER_TIMEOUT_SECONDS` | Open circuit cooldown | `120` |
| `CIRCUIT_BREAKER_PROBE_LOCK_SECONDS` | Half-open single-probe lock TTL | `10` |
| `MAX_RETRIES` | Per-service retry count | `3` |
| `RETRY_BACKOFF_BASE_SECONDS` | Exponential backoff base | `2` |
| `EXTERNAL_API_TIMEOUT_SECONDS` | External adapter HTTP timeout | `30.0` |
| `TASK_TIMEOUT_SECONDS` | Worker task watchdog | `60` |
| `PROCESSING_LOCK_TIMEOUT_SECONDS` | Stale processing reclaim age | `300` |
| `OUTBOX_POLL_INTERVAL_SECONDS` | Worker outbox poll interval | `1.0` |
| `WORKER_HEALTH_PORT` | Worker health server port | `8004` |
| `LOG_LEVEL` | Runtime logging level | `INFO` |

## Testing

Install dependencies locally:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the broad verification command used for the latest Phase 10 pass:

```bash
.venv/bin/pytest tests -q --cov=api --cov=engine --cov=ml --cov=services --cov=worker --cov-report=term
```

Latest result:

```text
187 passed
TOTAL 86%
```

Run only unit tests:

```bash
.venv/bin/python -m pytest tests/unit -q
```

Latest unit-only result:

```text
included in the full repository pass above
```

Run the official held-out `XGB_V1` evaluation report:

```bash
LENDING_CLUB_DATA_PATH="ml/data/raw/accepted_2007_to_2018Q4.csv.gz" \
  .venv/bin/python -m ml.models.evaluate --official-xgb-v1
```

Latest official held-out test metrics:

```text
AUC-ROC 0.975664
Brier 0.025293
ECE 0.003550
```

Run the official benchmark comparison:

```bash
LENDING_CLUB_DATA_PATH="ml/data/raw/accepted_2007_to_2018Q4.csv.gz" \
  .venv/bin/python -m ml.benchmark.heuristic_vs_ml --official-xgb-v1 --ml-threshold 0.5
```

Run integration and chaos tests with live Postgres/Redis:

```bash
docker compose up -d postgres redis credit-bureau bank-analyzer gst-verifier
.venv/bin/python -m pytest tests/integration tests/chaos -q
```

## Database and Data Safety Checks

Confirm encrypted application storage:

```bash
docker compose exec postgres psql -U auditlend -d auditlend \
  -c "SELECT pan_hash, encrypted_user_data IS NOT NULL AS has_ciphertext, encryption_nonce IS NOT NULL AS has_nonce FROM loan_applications LIMIT 5;"
```

Confirm audit log immutability:

```bash
docker compose exec postgres psql -U auditlend -d auditlend \
  -c "UPDATE audit_logs SET step = 'TEST' WHERE id = 1;"
```

Expected: PostgreSQL rejects the update with the append-only trigger.

## CI

GitHub Actions workflow:

```text
.github/workflows/ci.yaml
```

The workflow provisions Postgres and Redis, runs migrations, sets required crypto/auth env vars, runs tests with coverage, and fails if tests are skipped or coverage falls below the configured threshold.

## Rule Governance

Default heuristic rule set:

```text
RULE_SET_V1
```

ML-assisted decisions use:

```text
RULE_SET_V2
```

Rule sets are immutable dataclasses in `engine/rule_sets.py`. Changing weights or thresholds should create a new rule set version and update `docs/CALIBRATION.md`.

Current calibration status:

- `RULE_SET_V1` uses SME-derived heuristic weights.
- `RULE_SET_V2` is the ML-assisted scoring path activated only when ML is enabled or when the A/B assignment routes traffic to the ML arm.
- The ML stack is implemented, calibrated, benchmarked, fairness-scored, drift-monitored, and audit-linked using the official `XGB_V1` artifact set.

## Operational Notes

- The worker must register `worker.tasks.process_application.process_application`; this is verified in Docker logs during final testing.
- The outbox poller starts when the Celery worker is ready.
- API and worker images set `PYTHONPATH=/app` so imports are stable inside containers.
- The worker preloads `XGB_V1` from `ml/models/manifest.yaml` at startup and mounts the model directory read-only in Docker.
- The official drift snapshot lives at `ml/models/XGB_V1_reference_snapshot.json` and is used for non-blocking live KS-based alerting.
- `docker-compose.yml` does not run API with `--reload`; dev reload belongs in `docker-compose.dev.yml`.
- The local stack intentionally keeps infrastructure simple. Production deployment should move secrets, TLS, identity, logging retention, container hardening, and network policy into the target platform.

## Project Principles

- Determinism beats hidden randomness.
- Idempotency is part of correctness.
- Audit logs must be useful and PII-safe.
- Risk score, data reliability, and confidence are separate concepts.
- Fallback data should reduce confidence and route ambiguous cases to review.
- The explanation endpoint must explain what actually happened, using the audit trail.
- The README should describe the system as it is, not as a sales page.
