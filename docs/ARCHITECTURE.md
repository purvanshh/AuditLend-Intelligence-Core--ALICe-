# Architecture

## System Pipeline

```
Client
  |
  | POST /api/v1/apply-loan
  v
FastAPI API
  | \
  |  \-- PostgreSQL
  |       - loan_applications (encrypted PII)
  |       - idempotency_records
  |       - outbox
  |       - external_data
  |       - audit_logs (immutable, append-only)
  |
  \-- Redis
       - idempotency cache (fast path)
       - Celery broker / result backend
       - circuit breaker state

Celery Worker
  |
  |-- polls outbox
  |-- atomically claims applications
  |-- fetches / reuses external data
  |-- computes decision (heuristic or ML)
  |-- stores audit trail
  |
  |-- Credit Bureau Mock  (:8001)
  |-- Bank Analyzer Mock  (:8002)
  \-- GST Verifier Mock   (:8003)
```

## Pipeline Walkthrough

1. **`POST /api/v1/apply-loan`** validates input, checks idempotency (Redis → Postgres), encrypts PII (AES-256-GCM), writes application + outbox intent, returns immediately.

2. **Worker** polls the outbox, atomically claims the application (status → `PROCESSING`), reuses any already-persisted external data on retry.

3. **External data fetching** — credit bureau, bank analyzer, and GST verifier are called with retry/backoff, circuit breaker, and typed failure handling.

4. **Decision computation**:
   - **RULE_SET_V1**: Deterministic heuristic scorecard (credit + DTI + income stability + GST compliance).
   - **RULE_SET_V2**: ML-assisted path — XGB_V1 predicts default probability, isotonic calibration adjusts it, SHAP explains it, drift detector scores it. Falls back to V1 if confidence is low.

5. **Audit logging** — every step records an immutable audit entry. The explanation endpoint reads from the audit trail, not from live recomputation.

## Sequence Diagram (Decision Flow)

```
Apply → Idempotency Check → Encrypt PII → Write Application + Outbox
  → Worker Claims → Fetch External Data → Compute Risk Score
  → ML Scoring (if enabled) → Calibrate → SHAP Explain → Drift Check
  → Apply Rules → Store Decision → Append Audit Log
  → Return Decision
```

## Repository Structure

| Path | Purpose |
| --- | --- |
| `api/` | FastAPI app, auth (`X-API-Key`), routes, Pydantic schemas |
| `worker/` | Celery app, outbox poller, `process_application` task |
| `engine/` | `scoring.py` (risk score), `rule_sets.py` (immutable rules), `decision.py` (orchestration), `explanation.py` (audit trail reader) |
| `ml/models/` | `train.py`, `evaluate.py`, `calibrate.py`, model artifacts (`.pkl`), manifests |
| `ml/explain/` | `shap_explainer.py` — per-prediction SHAP contributions |
| `ml/governance/` | `model_registry.py`, `drift_detector.py`, `ab_test.py` |
| `ml/data/` | `ingestion.py`, `features.py`, `splits.py` — Lending Club pipeline |
| `ml/benchmark/` | Heuristic vs. ML comparison scripts and reports |
| `services/` | `credit_bureau.py`, `bank_analyzer.py`, `gst_verifier.py`, `crypto.py`, `audit_safety.py`, `metrics.py` |
| `models/` | SQLAlchemy models (`loan_application.py`, `audit_log.py`, etc.) |
| `mock_apis/` | Deterministic mock servers for bureau, bank, GST |
| `tests/` | Unit, integration, and chaos tests |
| `migrations/` | Alembic migration scripts |
| `research/` | EDA, WOE/IV, survival analysis, causal inference notebooks |
| `docs/` | Architecture, calibration, business impact, security, governance |

## API Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/api/v1/apply-loan` | write | Submit loan application |
| GET | `/api/v1/status/{id}` | read | Check application status |
| GET | `/api/v1/decision/{id}` | read | Get decision result |
| GET | `/api/v1/explanation/{id}` | read | Get human-readable explanation |
| GET | `/metrics` | none | Prometheus metrics |

### `POST /api/v1/apply-loan`

Headers: `Content-Type: application/json`, `X-API-Key`, `Idempotency-Key`

Request:
```json
{
  "idempotency_key": "req-001",
  "user_data": {
    "name": "Jane Doe", "pan": "ABCDE1234F",
    "monthly_income": 120000, "existing_emis": 25000,
    "loan_amount": 500000, "tenure_months": 36
  },
  "failure_flags": {
    "credit_bureau": "SUCCESS", "bank_analyzer": "SUCCESS", "gst_verifier": "SUCCESS"
  }
}
```

Responses: `201` (new), `200` (replay), `400` (validation), `401` (auth), `409` (idempotency conflict)

### `GET /api/v1/explanation/{id}`

Returns a human-readable summary with timeline, model factors, and SHAP contributions, derived from the audit trail.

## Security Architecture

- **PII Encryption**: AES-256-GCM for stored application data. 64-char hex key.
- **PAN Hashing**: Salted SHA-256. Raw PAN never stored in application rows.
- **API Authentication**: `X-API-Key` header with read/write scoping.
- **Audit Safety**: Database trigger prevents `UPDATE`/`DELETE` on `audit_logs`.
- **Input Sanitization**: Audit snapshots band raw financial values before persistence.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | Sync Postgres URL | `postgresql://auditlend:auditlend@postgres:5432/auditlend` |
| `ASYNC_DATABASE_URL` | Async Postgres URL | `postgresql+asyncpg://...` |
| `REDIS_URL` | Celery + cache + circuit breaker | `redis://redis:6379/0` |
| `PII_ENCRYPTION_KEY` | 64-char hex AES-256-GCM key | (required) |
| `PAN_HASH_SALT` | Per-environment PAN salt | (required) |
| `CONFIDENCE_THRESHOLD` | Manual review threshold | `0.6` |
| `API_KEYS` | Comma-separated key:scope pairs | (dev-only in override) |
| `CIRCUIT_BREAKER_THRESHOLD` | Failures before open | `5` |
| `CIRCUIT_BREAKER_WINDOW_SECONDS` | Counting window | `60` |
| `MAX_RETRIES` | Per-service retry count | `3` |
| `RETRY_BACKOFF_BASE_SECONDS` | Exponential backoff base | `2` |
| `EXTERNAL_API_TIMEOUT_SECONDS` | HTTP timeout for providers | `30.0` |

## Infrastructure

- **PostgreSQL**: Source of truth for applications, decisions, audit logs, idempotency.
- **Redis**: Idempotency cache (fast path), Celery broker/backend, circuit breaker state.
- **Celery**: Async task processing with outbox pattern for reliable delivery.
- **Prometheus Metrics**: Application counts, external API latency, circuit breaker state, decision confidence, task duration, drift alerts, A/B assignments.
- **Docker Compose**: Local/demo deployment. No production orchestrator.

## ML Stack

- **Training**: XGBoost on Lending Club (2007–2016 train, 2017 validation, 2018 test).
- **Features**: 38 engineered features (credit utilization, DTI, payment history, account density).
- **Calibration**: Isotonic regression on validation split. ECE reduced from 0.016 to 0.0036.
- **Explainability**: SHAP TreeExplainer — top-5 per-prediction feature contributions.
- **Drift Detection**: KS-test against training reference snapshot. Advisory alerts.
- **Model Registry**: File-backed (YAML manifest). Versioned artifacts (model, calibrator, features, snapshot).
- **A/B Testing**: Deterministic routing based on grade + purpose hash. Metrics via Prometheus.
