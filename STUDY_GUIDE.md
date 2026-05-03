# AuditLend Intelligence Core (ALICe) — Comprehensive Study Guide

This document provides a deep architectural and implementation-level analysis of AuditLend Intelligence Core, a deterministic, idempotent, audit-grade credit decision engine. The guide assumes the reader has working knowledge of Python, SQL, REST APIs, and distributed systems fundamentals. Each section builds from first principles to practical implementation details.

---

## 1. Project Overview

### 1.1 What the Project Does (Practical Terms)

AuditLend Intelligence Core is a loan application processing system that receives credit applications, fetches data from external providers (credit bureau, bank analyzer, GST verifier), computes a risk score using either a weighted heuristic scorecard or an ML model (XGBoost), applies immutable rule sets to decide whether to approve, decline, or route an application to manual review, and produces a human-readable explanation derived entirely from the audit trail. Every decision step is recorded in an append-only audit log that cannot be updated or deleted.

The system is built to survive worker crashes, duplicate task deliveries, external service outages, and regulatory scrutiny. It is not a production lending system — it uses deterministic mock providers rather than real credit bureaus — but the architecture and operational patterns are production-grade.

### 1.2 Core Problem It Solves

Lending decisions must be:
- **Deterministic**: identical inputs always produce identical outputs. No randomness anywhere in the business logic.
- **Idempotent**: network retries, worker restarts, and duplicate Celery task deliveries must not create duplicate decisions or violate once-only semantics.
- **Auditable**: every input, every external call, every intermediate computation, and every fallback must be captured in an immutable trail that can be reproduced for regulatory review.
- **Resilient**: external provider timeouts and failures must be classified, retried with backoff, and have circuit breakers that prevent cascading failures. When providers fail, conservative fallbacks must be applied and confidence must be reduced.
- **Explicable**: the explanation endpoint must describe what actually happened at decision time, not recompute from current business logic.

### 1.3 Key Features and Capabilities

| Feature | Implementation |
| --- | --- |
| Loan application intake | FastAPI `POST /api/v1/apply-loan` with JSON payload, returns immediately after write |
| Asynchronous processing | Celery worker polls outbox, atomically claims applications, fetches external data, computes decision |
| Idempotency | Redis fast path + PostgreSQL durable fallback with payload hash verification |
| PII protection | AES-256-GCM encryption for user data at rest; salted SHA-256 PAN hash (raw PAN never stored) |
| Append-only audit | SQLAlchemy models + PostgreSQL trigger (enforced at DB level) blocking UPDATE/DELETE |
| Heuristic risk scoring | Weighted 0-100 score computed in `engine/scoring.py`. Credit, income stability, DTI, GST compliance, data quality penalty |
| ML risk scoring | XGBoost model (XGB_V1) with calibrated probability, SHAP explanations, drift detection |
| Data reliability | Separate from risk score. Penalties applied per `FailureType` in `engine/confidence.py` |
| Calibrated confidence | `data_reliability * boundary_distance_factor` where factor depends on how far the risk score is from the threshold |
| Decision rules | Priority-ordered rule matching in `engine/rules.py`. GST non-compliance gates automatic approval. |
| External service resilience | Retry with exponential backoff, circuit breaker (Redis-backed), typed `FailureType` classification |
| Human-readable explanations | Built from audit log entries in `engine/explanation_builder.py`, not recomputed |
| A/B experimentation | Deterministic arm assignment via `ml/governance/ab_test.py` |
| Drift detection | KS-based live feature drift vs. reference snapshot, non-blocking alert in audit |

---

## 2. High-Level Architecture

### 2.1 Overall System Design

```
Client
  |
  | POST /api/v1/apply-loan
  v
FastAPI API (async)
  | \
  |  \-- PostgreSQL (ACID source of truth)
  |       - loan_applications (encrypted PII)
  |       - idempotency_records (request/response pairs)
  |       - outbox (transactional outbox pattern)
  |       - external_data (cached provider responses)
  |       - audit_logs (append-only)
  |
  \-- Redis (in-memory cache/broker)
       - idempotency cache (fast path)
       - Celery broker/result backend
       - circuit breaker state per external service
       |
       v
Celery Worker
       |-- polls outbox for PENDING messages
       |-- atomically claims with UPDATE ... WHERE status=PENDING
       |-- reuses persisted external data on retry
       |-- computes decision (heuristic or ML)
       |-- writes audit trail
       |
       v (external calls)
Credit Bureau Mock  |  Bank Analyzer Mock  |  GST Verifier Mock
     (port 8001)            (port 8002)              (port 8003)
```

This is a **layered async architecture**: the API layer is synchronous-to-async (writes to PostgreSQL and Redis, returns), the worker layer is entirely async (Celery task), and all external calls are async via `httpx.AsyncClient`. The database is the source of truth for all state; Redis is a cache layer only.

### 2.2 Major Components and How They Interact

| Component | Responsibility | Key Files |
| --- | --- | --- |
| `api/` | HTTP request handling, authentication, validation, idempotency check | `api/main.py`, `api/routes/applications.py`, `api/auth.py` |
| `worker/` | Celery app, outbox poller, task definition, atomic claiming | `worker/celery_app.py`, `worker/tasks/process_application.py`, `worker/outbox_poller.py` |
| `engine/` | Pure risk scoring, decision evaluation, confidence calibration, explanation building | `engine/scoring.py`, `engine/rules.py`, `engine/confidence.py`, `engine/decision.py`, `engine/explanation_builder.py` |
| `services/` | External service clients (retry, circuit breaker), PII crypto, audit writes, metrics | `services/base.py`, `services/crypto.py`, `services/audit.py`, `services/metrics.py` |
| `models/` | SQLAlchemy ORM models | `models/application.py`, `models/audit_log.py`, `models/idempotency.py`, `models/outbox.py`, `models/external_data.py` |
| `mock_apis/` | Deterministic test doubles for external providers | `mock_apis/credit_bureau.py`, `mock_apis/bank_analyzer.py`, `mock_apis/gst_verifier.py` |
| `ml/` | ML data ingestion, model training, calibration, SHAP explanations, drift detection, benchmarking | `ml/data/`, `ml/models/`, `ml/explain/`, `ml/governance/`, `ml/benchmark/` |

### 2.3 Data Flow Across the System

1. **Client submits** `POST /api/v1/apply-loan` with JSON payload containing `user_data`, optional `failure_flags` for testing, and an `Idempotency-Key` header.
2. **API validates** payload, computes `payload_hash = SHA256(payload + idempotency_key)`.
3. **API checks idempotency**:
   - Redis fast path: if key exists and hash matches, return cached response (HTTP 200).
   - PostgreSQL fallback: if key exists in `idempotency_records` and hash matches, cache to Redis and return (HTTP 200).
   - If key exists but hash differs, return HTTP 409 Conflict.
4. **API encrypts PII**: `PIIService.encrypt(user_data)` → AES-256-GCM ciphertext + nonce. Computes `PAN hash = SHA256(pan + salt)`.
5. **API writes** to `loan_applications` (status = PENDING), writes to `outbox` (task_name, task_args), writes to `idempotency_records`.
6. **API returns** HTTP 201 with `application_id`, status = PENDING. The response is committed in a single transaction with the outbox message (transactional outbox pattern).
7. **Celery worker polls** outbox for PENDING messages (via outbox poller, not the Celery task being invoked directly — the task is dispatched via the outbox, not via a direct Celery `apply_async`).
8. **Worker atomically claims** application: `UPDATE loan_applications SET status = 'PROCESSING' WHERE id = :id AND (status = 'PENDING' OR (status = 'PROCESSING' AND updated_at < stale_before))`. If claiming fails because the application is already COMPLETED, return the stored result. If it's PROCESSING by another worker, return a "being processed" response.
9. **Worker fetches external data** concurrently via `asyncio.gather`:
   - Calls credit bureau, bank analyzer, GST verifier via `BaseExternalService.call`.
   - Before each call, checks if `ExternalData` already exists for this `(application_id, source_type)` — if so, reuse the persisted result (external data reuse).
10. **External service client** (`services/base.py`):
    - Checks circuit breaker state in Redis. If OPEN, short-circuits and returns fallback.
    - Retries with exponential backoff + deterministic jitter for retryable failures (`TIMEOUT`, `SERVICE_DOWN`).
    - Classifies response into `FailureType`.
    - Updates circuit breaker state on success/failure.
11. **Worker computes decision** via `engine/decision.compute_decision_from_env`:
    - Extracts `credit_score`, `income_stability`, `gst_compliant`.
    - Computes heuristic risk score via `engine/scoring.compute_risk_score`.
    - If `ML_ENABLED=true`, loads XGBoost model, computes ML risk score, runs SHAP explanation.
    - Determines scoring strategy (heuristic vs. ML) and rule version.
    - Evaluates decision rules in priority order.
    - Computes data reliability and calibrated confidence.
    - If confidence < threshold, overrides decision to `NEEDS_REVIEW`.
12. **Worker stores results**:
    - Updates `loan_applications` with status, decision, confidence.
    - Writes `ExternalData` records for each provider (if not already written).
    - Writes audit log entries for every step: `PROCESSING_STARTED`, `{SOURCE}_FETCH`, `ML_SCORING` (if applicable), `DECISION_CALCULATION`, `MANUAL_REVIEW_OVERRIDE` (if applicable).
13. **Client queries**:
    - `GET /api/v1/status/{application_id}` returns current status.
    - `GET /api/v1/decision/{application_id}` returns decision, risk score, data reliability, confidence, rule version, model details.
    - `GET /api/v1/explanation/{application_id}` builds explanation from audit logs via `engine/explanation_builder`.

---

## 3. Why This Architecture?

### 3.1 Why This Architecture Was Chosen Over Alternatives

| Alternative | Why Rejected | Why AuditLend Chose the Current Architecture |
| --- | --- | --- |
| Synchronous HTTP call from API to providers | Blocks the API request thread; external failures cascade to API availability; no audit trail for external calls; no retry/backoff control | Async worker fetches external data via Celery; external calls are isolated from API latency; retry/backoff and circuit breaker live in the worker; full audit trail |
| Direct database polling by workers | Workers compete for rows, no atomic claiming, duplicate processing hard to prevent | Transactional outbox pattern: API writes intent to outbox table atomically with the application; worker polls outbox and atomically claims via `UPDATE ... WHERE status=PENDING`. This ensures exactly-once processing even with multiple workers. |
| In-memory idempotency | Lost on restart or across workers; no durability guarantee | Redis fast path + PostgreSQL durable fallback. Redis is checked first for latency, but PostgreSQL is the durable source of truth. |
| Event sourcing | Over-engineered for a loan processing pipeline; adds complexity without benefit in this domain | Append-only audit log table captures every step. This is simpler than event sourcing but provides the same immutability guarantee. |
| Inline ML in API | Blocks API thread; ML model loading is slow; no graceful fallback on model unavailability; harder to govern | ML runs in the worker. The worker preloads the model at startup if `ML_ENABLED=true`. If the model fails to load or confidence is low, the worker deterministically falls back to the heuristic scorecard. |
| Updateable audit logs | Regulatory violation; can be tampered with; defeats the purpose of an audit trail | Append-only audit table enforced at the database level via PostgreSQL trigger (`migrations/versions/20260429_0005_audit_protection.py`). UPDATE/DELETE raises an exception. |

### 3.2 Trade-Offs

| Trade-off | Impact | Mitigation |
| --- | --- | --- |
| Async processing adds latency | Client cannot get decision in the initial POST response; must poll `/status` and `/decision` | The README documents the expected latency (~1–2 seconds for the happy path). The API returns immediately, which is a feature for UX. |
| Two-tier idempotency (Redis + Postgres) | Race conditions possible if Redis has stale data after a Postgres write | Always verify `payload_hash` before trusting Redis. PostgreSQL is the authoritative source. |
| Celery worker failure handling | Poison messages can block the queue | `max_retries=0` in task definition; task catches exceptions and marks application as MANUAL_REVIEW. No retry, but audit trail is preserved. |
| Encryption at rest | Requires key management; if key is lost, data is unrecoverable | Per-deployment key generation via environment variable; documented in README; keys are not in the code. |
| Deterministic mocks in testing | Not a real provider integration; may not catch provider-specific bugs | The mocks are deterministic by design (same input = same output) to allow reproducible testing. Real provider integration would require another system. |

### 3.3 When This Architecture Fails or Becomes Inefficient

- **Very low latency requirements**: The async pipeline adds 1–2 seconds minimum. For sub-second loan decisions, this architecture is not suitable without major redesign (inline scoring, synchronous external calls).
- **Very high throughput**: The single PostgreSQL writer becomes a bottleneck above ~100 applications/second. Read replicas and a message queue (Kafka) would be needed at scale.
- **Very long external calls**: If a provider takes >30 seconds, the task timeout (60 seconds) will trigger and the application will be marked MANUAL_REVIEW. Increase `TASK_TIMEOUT_SECONDS` or add a longer timeout / async callback pattern.
- **Multi-region deployment**: The transactional outbox pattern assumes a single PostgreSQL instance. Multi-region would require distributed transaction coordination (e.g.,Saga pattern), which is not implemented.
- **Real-time ML scoring**: The worker preloads the ML model at startup. If the model file is large (>100MB), container startup time increases significantly. The model could be loaded on-demand with a cache, but that's not implemented.

---

## 4. Tech Stack Breakdown

### 4.1 Languages, Frameworks, Libraries, Databases

| Layer | Technology | Version (from requirements files; see README for the exact list) |
| --- | --- | --- |
| Language | Python | 3.11+ |
| API framework | FastAPI | async-first, Pydantic for validation |
| Task queue | Celery | async task execution, Redis broker |
| Database | PostgreSQL | 16-alpine (from docker-compose.yml) |
| Cache / broker | Redis | 7-alpine (from docker-compose.yml) |
| ORM | SQLAlchemy | 2.x async dialect (asyncpg driver) |
| HTTP client | httpx | async HTTP for external service calls |
| Encryption | cryptography (pyca/cryptography) | AES-256-GCM for PII |
| Hashing | hashlib (stdlib) | SHA-256 for PAN hash |
| Logging | structlog | structured JSON logging |
| Metrics | prometheus_client | Prometheus export via `/metrics` |
| ML framework | XGBoost, scikit-learn, shap | XGB_V1 model, isotonic calibration, SHAP explanations |
| Migration tool | Alembic | database schema migrations |

### 4.2 Why Each Was Likely Chosen

| Technology | Rationale |
| --- | --- |
| **FastAPI** | Native async support, automatic OpenAPI docs, Pydantic validation, middleware support, and good performance. Starlette under the hood. |
| **Celery** | Mature task queue with Redis broker, atomic task claiming, result backend, retry policies, andFlower monitoring UI. Integrates well with PostgreSQL and Redis. |
| **PostgreSQL** | ACID transactions are essential for the transactional outbox pattern and audit log immutability. JSONB columns for flexible external data storage. UUID type for application IDs. |
| **Redis** | Low-latency cache for idempotency, circuit breaker state, Celery broker/result backend. In-memory, single-threaded, fast. |
| **SQLAlchemy 2.x** | Strong typing, async dialect, and ORM that maps cleanly to the audit log and application models. |
| **httpx** | Async HTTP client required for concurrent external service calls in the worker. Synchronous `requests` would block the event loop. |
| **cryptography (AESGCM)** | Industry-standard AEAD encryption. AES-256-GCM provides both confidentiality and integrity. The library is maintained and audited. |
| **structlog** | Structured logging (JSON) simplifies log aggregation in containerized environments. Key-value pairs in logs are easier to query than text logs. |
| **prometheus_client** | Standard Prometheus metrics export. `/metrics` endpoint is scraped by Prometheus. Histograms for latency, counters for throughput. |
| **XGBoost** | Gradient boosting is the standard for tabular credit risk models. High performance, good calibration with isotonic regression. |
| **SHAP** | Model-agnostic explainability. Required for the "human-readable explanations" requirement. |
| **Alembic** | SQLAlchemy's migration tool. Supports multi-environment deployments and schema versioning. |

### 4.3 What Alternatives Could Have Been Used and Why They Weren't

| Alternative | Why It Could Have Been Used | Why It Wasn't |
| --- | --- | --- |
| **Django + Django ORM** | Faster development, built-in auth, admin UI | Over-engineered for a Loan processing API; adds latency and memory footprint; Celery integration is simpler with FastAPI |
| **SQLAlchemy 1.x** | Stable, widely used | Async support is clunky; 2.x has native async |
| **aiohttp** | Async HTTP server in stdlib | Less feature-rich than FastAPI (validation, docs, middleware) |
| **Redis Streams** (instead of Celery) | Native long-polling support, simpler stack | Celery provides task deduplication, retry policies, result backend, and Flower monitoring out of the box |
| **LightGBM** | Faster training than XGBoost on large datasets | XGBoost is the standard in Lending Club benchmarks; the project used XGBoost in the official evaluation |
| **CatBoost** | Handles categorical features natively | Not needed for the feature engineering in this project |
| **Kafka** (instead of PostgreSQL outbox) | Higher throughput, log retention | Over-engineered for the volume; PostgreSQL is sufficient and simplifies the stack |

---

## 5. Folder & Code Structure Deep Dive

### 5.1 Explain Each Major Folder/Module

```
/Users/purvansh/Desktop/Projects/AuditLend Intelligence Core (ALICe)/
├── api/                          # FastAPI application, routes, schemas
│   ├── main.py                  # FastAPI app setup, middleware, CORS, health endpoint
│   ├── auth.py                 # API key authentication (X-API-Key header)
│   ├── dependencies.py         # Async session dependency for SQLAlchemy
│   ├── routes/
│   │   ├── applications.py     # POST /apply-loan, GET /status/{id}
│   │   ├── decisions.py        # GET /decision/{id}
│   │   └── explanations.py    # GET /explanation/{id}
│   └── schemas/
│       ├── application.py      # Pydantic request/response models for loan applications
│       ├── decision.py        # Pydantic models for decision output
│       └── explanation.py     # Pydantic models for explanation output
├── worker/                     # Celery application and task definitions
│   ├── celery_app.py         # Celery app instantiation, task registry
│   ├── tasks/
│   │   └── process_application.py  # Main async task: fetch external data, compute decision, write audit
│   └── outbox_poller.py      # Background poller (optional; Celery can invoke tasks directly)
├── engine/                     # Pure decision engine (no I/O, no network, no database, no randomness)
│   ├── scoring.py            # compute_risk_score() — weighted 0-100 heuristic; MLScorer class
│   ├── rule_sets.py         # Immutable RuleSet dataclasses (RULE_SET_V1, RULE_SET_V2)
│   ├── rules.py            # evaluate() — priority-ordered decision rule matching
│   ├── confidence.py       # compute_data_reliability(), compute_decision_confidence()
│   ├── decision.py         # compute_decision() — orchestrates all the above
│   └── explanation_builder.py  # build_explanation() — reconstructs human-readable explanation from audit logs
├── services/                  # External service clients, crypto, audit writes, logging
│   ├── __init__.py        # FailureType enum, ServiceResult dataclass
│   ├── base.py            # BaseExternalService class with retry, circuit breaker, metrics
│   ├── credit_bureau.py  # CreditBureauService extending BaseExternalService
│   ├── bank_analyzer.py   # BankAnalyzerService extending BaseExternalService
│   ├── gst_verifier.py    # GstVerifierService extending BaseExternalService
│   ├── crypto.py         # PIIService: AES-256-GCM encryption, salted SHA-256 PAN hash
│   ├── audit.py          # write_audit_entry() — append-only to audit_logs table
│   ├── metrics.py        # Prometheus metric definitions
│   └── logging.py        # structlog configuration
├── models/                    # SQLAlchemy ORM models
│   ├── application.py     # LoanApplication
│   ├── audit_log.py      # AuditLog (APPEND ONLY)
│   ├── idempotency.py    # IdempotencyRecord
│   ├── outbox.py         # OutboxMessage (transactional outbox)
│   ├── external_data.py # ExternalData (cached provider responses)
│   └── __init__.py       # Exports all models
├── migrations/              # Alembic migrations
│   ├── versions/
│   │   ├── 20260429_0001_initial.py    # Initial schema
│   │   ├── 20260429_0002_encrypt_user_data.py  # Add encrypted PII columns
│   │   ├── 20260429_0003_enforce_encryption.py  # NOT NULL constraints on encrypted columns
│   │   ├── 20260429_0004_outbox_and_external_idempotency.py  # Outbox and ExternalData tables
│   │   └── 20260429_0005_audit_protection.py   # Trigger to block UPDATE/DELETE on audit_logs
│   └── env.py            # Alembic environment configuration
├── mock_apis/               # Deterministic test doubles for external providers
│   ├── credit_bureau.py   # Mock credit bureau with SUCCESS/TIMEOUT/STALE_DATA/SERVICE_DOWN
│   ├── bank_analyzer.py  # Mock bank analyzer with SUCCESS/PARTIAL_DATA/FORMAT_ERROR
│   ├── gst_verifier.py   # Mock GST verifier with SUCCESS/PAN_MISMATCH/NO_RECORD
│   └── run_all.py       # Helper to run all mocks
├── ml/                     # ML platform (data, models, explainability, governance, benchmark)
│   ├── data/              # Lending Club data ingestion, feature engineering
│   ├── models/            # XGB_V1 training, isotonic calibration, manifest
│   ├── explain/           # SHAP explainer
│   ├── governance/       # Model registry, drift detection, A/B experiment assignment
│   └── benchmark/        # Heuristic vs. ML comparison script
├── tests/                  # Test suite
│   ├── unit/             # Unit tests (no external dependencies)
│   ├── integration/      # Integration tests (PostgreSQL + Redis required)
│   └── chaos/            # Chaos tests (circuit breaker, idempotency under load, retry exhaustion)
├── docs/
│   └── CALIBRATION.md    # Calibration notes, heuristic scorecard vs. ML benchmark results
├── docker-compose.yml     # Full stack: postgres, redis, 3 mocks, api, worker, flower
├── Dockerfile.api        # API container
├── Dockerfile.worker    # Worker container (includes ML artifacts mount)
├── Dockerfile.mock       # Mock provider container
└── README.md            # This is the source of truth for the project
```

### 5.2 Responsibility of Each Component

| Component | Responsibility |
| --- | --- |
| `api/main.py` | FastAPI app factory, middleware setup, CORS, health endpoint, exception handlers |
| `api/routes/applications.py` | Idempotency check (Redis fast path, PostgreSQL fallback), PII encryption, transactional write of application + outbox + idempotency record, response handling |
| `api/routes/decisions.py` | GET /decision/{id} — retrieves decision from loan_applications, expands with rule version, model details |
| `api/routes/explanations.py` | GET /explanation/{id} — fetches audit logs, builds human-readable explanation via explanation_builder |
| `engine/scoring.py` | Pure function `compute_risk_score(credit_score, income_stability, dti, gst_compliant, failure_types, rule_set)`. MLScorer class wraps XGBoost model loading, feature mapping, SHAP explanation, drift detection |
| `engine/rule_sets.py` | Immutable RuleSet dataclasses. `RULE_SET_V1` is the baseline. `RULE_SET_V2` is the ML-assisted path. Changing weights or thresholds requires a new RuleSet version |
| `engine/rules.py` | `evaluate()` — pure function that applies decision rules in priority order. First matching rule wins. GST non-compliance caps the effective risk score |
| `engine/confidence.py` | `compute_data_reliability()` — base reliability 1.0 minus penalties per FailureType. `compute_decision_confidence()` — data_reliability * boundary_distance_factor |
| `engine/decision.py` | `compute_decision()` orchestrates extraction, scoring, confidence, ML (if enabled), and rule evaluation. Returns DecisionOutput dataclass |
| `engine/explanation_builder.py` | `build_explanation()` reads audit entries and constructs a human-readable summary, factor objects, timeline, and model factor contributions |
| `services/base.py` | BaseExternalService with async HTTP, retry with exponential backoff + deterministic jitter, typed FailureType classification, Redis-backed circuit breaker |
| `services/crypto.py` | PIIService wrapping AESGCM and SHA-256. Encrypts entire user_data payload, not individual fields. Hashes PAN with per-deployment salt |
| `services/audit.py` | `write_audit_entry()` — appends to audit_logs. Called at every decision step in the worker |
| `services/metrics.py` | Prometheus Gauge/Counter/Histogram definitions |
| `worker/tasks/process_application.py` | Main Celery task. Claims application atomically via UPDATE, fetches external data with reuse, computes decision, stores results, writes audit entries |
| `models/application.py` | LoanApplication ORM model. Stores encrypted_user_data, encryption_nonce (ciphertext), pan_hash (not raw PAN), status, decision, confidence |
| `models/audit_log.py` | AuditLog ORM model. Step, input_snapshot, output_snapshot, error_type, fallback_used, rule_version. UPDATE/DELETE blocked by DB trigger |
| `models/idempotency.py` | IdempotencyRecord. Stores request payload hash and response for replay |
| `models/outbox.py` | OutboxMessage. Transactional outbox entry. Worker polls this table for PENDING messages |
| `models/external_data.py` | ExternalData. Caches provider responses to avoid redundant calls on worker retry |

### 5.3 How Modules Are Connected

- **API → Worker**: Application ID is written to the outbox table. The worker polls that table (not Celery's task invocation directly). This is the transactional outbox pattern.
- **API → Idempotency**: Client sends `Idempotency-Key` header. API computes a SHA256 hash of the full JSON payload + idempotency key. If a matching key exists with a matching hash, the API returns the cached response. If the key exists but the hash differs (different payload), the API returns 409 Conflict.
- **API → Encryption**: PII is encrypted before being written to the database. The worker decrypts it when processing.
- **Worker → External services**: `BaseExternalService.call()` is the abstract base. Concrete services (credit_bureau, bank_analyzer, gst_verifier) extend it and implement `retryable_failures()`, `classify_response()`, and the API endpoint path.
- **Worker → Engine**: The worker's main task calls `compute_decision_from_env()`. All computation in the engine is pure (no I/O, no randomness). The worker passes in the external service results and user data.
- **Worker → Audit**: Every significant step in the worker calls `write_audit_entry()`. The audit log entries are used by the explanation builder to reconstruct what happened.
- **Engine → Scoring**: `compute_risk_score()` computes the heuristic score. `MLScorer.score()` computes the ML score (if enabled). Both produce a risk score in the 0–100 range.
- **Engine → Confidence**: `compute_data_reliability()` computes how reliable the external data is (based on failure types). `compute_decision_confidence()` multiplies data_reliability by a boundary distance factor.
- **Engine → Rules**: `evaluate()` applies the decision rules based on the risk score, DTI, and GST compliance.
- **Explanation → Audit**: The explanation builder does not recompute anything. It reads the audit log entries that were written at decision time and templates them into human-readable text.

---

## 6. Core Workflows

### 6.1 Step-by-Step Execution of the Happy Path

This section traces a single loan application from submission to decision.

#### 6.1.1 Client Submits Application

```bash
curl -X POST http://localhost:8000/api/v1/apply-loan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-read-write" \
  -H "Idempotency-Key: happy-001" \
  -d '{
    "idempotency_key": "happy-001",
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
  }'
```

**What happens inside the API:**
1. `apply_loan()` in `api/routes/applications.py` receives the request.
2. Pydantic validates the schema (`ApplyLoanRequest`).
3. `require_auth()` middleware validates the `X-API-Key` header.
4. `_payload_hash(request, key)` computes `SHA256(serialize(request) + key)`.
5. Redis is queried for `idempotent:{key}`. If found and hash matches, return cached response (HTTP 200).
6. PostgreSQL is queried for `IdempotencyRecord` with `key`. If found and hash matches, cache to Redis and return (HTTP 200).
7. If key is new, `pii_service.encrypt(user_data)` produces `(ciphertext, nonce)`.
8. `pii_service.hash_pan(pan)` produces the salted SHA-256 hash.
9. A new `LoanApplication` is created with the encrypted PII, status = "PENDING".
10. A new `OutboxMessage` is created with `task_name = "worker.tasks.process_application.process_application"` and `task_args = {"application_id": "..."}`.
11. A new `IdempotencyRecord` is created with the payload hash and response.
12. All three inserts are in the same database transaction (commit on success).
13. The response is cached to Redis with TTL `IDEMPOTENCY_CACHE_TTL_SECONDS` (86400 seconds).
14. API returns HTTP 201 with `{"application_id": "...", "status": "PENDING", "message": "Application received and queued for processing"}`.
15. `loan_applications_total.labels(status="PENDING").inc()` is recorded.

#### 6.1.2 Worker Processes Application

The Celery worker (running `celery -A worker.celery_app worker --loglevel=info --concurrency=2`) polls the outbox. In the Docker Compose setup, the outbox poller is not used; instead, the task is dispatched via `apply_async` when the outbox is written. The task `process_application` is invoked with the `application_id`.

**Inside `worker/tasks/process_application.py`:**

1. **`_claim_application(application_id)`** runs an atomic claim:
   ```sql
   UPDATE loan_applications
   SET status = 'PROCESSING', updated_at = NOW()
   WHERE id = :id
     AND (status = 'PENDING'
          OR (status = 'PROCESSING' AND updated_at < :stale_before))
   RETURNING id
   ```
   - If `claimed_id` is None and the existing status is COMPLETED or MANUAL_REVIEW, return the stored result (idempotent completion).
   - If `claimed_id` is None but status is PROCESSING (being processed by another worker), return a "being processed" response.
   - Otherwise, the claim succeeds. Write an audit entry: `step = "PROCESSING_STARTED"`.

2. **`_fetch_external_data(application_id, user_data, failure_flags, redis_client)`** runs three concurrent HTTP calls:
   ```python
   credit_result, bank_result, gst_result = await asyncio.gather(
       _fetch_or_reuse_external_data(..., credit_service.fetch(...)),
       _fetch_or_reuse_external_data(..., bank_service.analyze(...)),
       _fetch_or_reuse_external_data(..., gst_service.verify(...)),
   )
   ```
   - Each `_fetch_or_reuse_external_data()` check if `ExternalData` already exists for this `(application_id, source_type)`. If yes, reuse it (external data reuse on worker retry). If no, call the provider and store the result.

3. **Inside each service call (`services/base.py` `BaseExternalService.call()`):**
   - Check Redis for circuit breaker state. If OPEN, return immediately with `FailureType.SERVICE_DOWN` and `fallback_used=True`.
   - Make up to `MAX_RETRIES` (3) attempts with exponential backoff:
     ```python
     delay = RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt) + deterministic_jitter
     ```
   - The deterministic jitter prevents thundering herd but is reproducible: `((attempt + 1) * 137 % 500) / 1000`.
   - If the response is successful, classify it as `FailureType.SUCCESS`. Update circuit breaker to CLOSED.
   - If the response is a retryable failure (TIMEOUT, SERVICE_DOWN), retry. After exhausting retries, record the failure and update circuit breaker state (if failures >= `CIRCUIT_BREAKER_THRESHOLD`, open the circuit).
   - If the response is a non-retryable failure (PARTIAL_DATA, FORMAT_ERROR, PAN_MISMATCH, NO_RECORD), do not retry; record the failure type and return.

4. **`compute_decision_from_env(credit_result, bank_result, gst_result, decision_user_data, failure_flags, application_id)`** runs in `engine/decision.py`:
   - Extract `credit_score`, `income_stability`, `gst_compliant` from the service results.
   - Compute DTI = `existing_emis / monthly_income`.
   - Run heuristic scoring:
     ```python
     heuristic_risk_score, breakdown = compute_risk_score(
         credit_score, income_stability, dti, gst_compliant, failure_types, rule_set
     )
     ```
     where `compute_risk_score()` applies the weighted formula:
     ```python
     score = (credit_component) + (stability_component) + (dti_component) + (gst_component) - (data_quality_penalty)
     risk_score = clamp(score, 0.0, 100.0)
     ```
   - Run ML scoring if `ML_ENABLED=true` (in `engine/scoring.py`):
     - Load XGBoost model from `ML_MODEL_MANIFEST_PATH`.
     - Map external data to ML feature row (`_build_ml_feature_row()`).
     - Run SHAP explanation (`explain_feature_row()`).
     - Compute calibrated probability from isotonic calibrator.
     - If model confidence < `CONFIDENCE_THRESHOLD`, set `fallback_used = True` and use heuristic score instead.
   - Compute data reliability:
     ```python
     data_reliability, penalty_reasons = compute_data_reliability(failure_types, used_fallback_credit)
     ```
     where penalties are:
     ```python
     PENALTIES = {
         FailureType.TIMEOUT: 0.30,
         FailureType.STALE_DATA: 0.20,
         FailureType.SERVICE_DOWN: 0.30,
         FailureType.PARTIAL_DATA: 0.20,
         FailureType.FORMAT_ERROR: 0.30,
         FailureType.PAN_MISMATCH: 0.20,
         FailureType.NO_RECORD: 0.10,
     }
     ```
   - Compute calibrated confidence:
     ```python
     confidence, reasons = compute_decision_confidence(risk_score, decision, data_reliability, failure_types)
     confidence = data_reliability * boundary_distance_factor(risk_score, decision)
     ```
     where the boundary factor is:
     ```python
     # APPROVE: risk_score >= 80 → 1.0, >= 70 → 0.9, >= 55 → 0.7, < 55 → 0.6
     # DECLINE: risk_score <= 20 → 1.0, <= 34 → 0.85, > 34 → 0.75
     # NEEDS_REVIEW: 0.5
     ```
   - Evaluate decision rules:
     ```python
     decision, factors = evaluate(risk_score, credit_score, dti, failure_types, gst_compliant, rule_set)
     # Rule 1: risk_score >= 70 and no failures → APPROVE
     # Rule 2: risk_score >= 55 and dti < 0.5 → APPROVE
     # Rule 3: risk_score < 35 or dti > 0.6 → DECLINE
     # Rule 4: otherwise → NEEDS_REVIEW
     # GST gate: if gst_compliant is False, cap effective_risk_score at (approve_moderate_threshold - 1.0)
     ```
   - If confidence < `CONFIDENCE_THRESHOLD` (0.6), override decision to `NEEDS_REVIEW`.

5. **`_store_processing_results()`** writes to the database:
   - Update `loan_applications` with status, decision, confidence.
   - Write audit entries:
     - `ML_SCORING` (if ML was attempted)
     - `DRIFT_DETECTED` (if drift alerts > 0)
     - `DECISION_CALCULATION` with the full decision output
     - `MANUAL_REVIEW_OVERRIDE` (if confidence below threshold)

6. The task returns a dictionary:
   ```python
   {
       "application_id": str(app_id),
       "status": _status_for_decision(decision_output),
       "decision": decision_output.decision.value,
       "confidence": decision_output.confidence,
       "data_reliability": decision_output.data_reliability,
       "risk_score": decision_output.risk_score,
       "rule_version": decision_output.rule_version,
       "model_version": decision_output.model_version,
       "scoring_strategy": decision_output.scoring_strategy,
       "ab_test_arm": decision_output.ab_test_arm,
   }
   ```

#### 6.1.3 Client Polls for Decision

```bash
curl -s -H "X-API-Key: dev-key-read-only" \
  http://localhost:8000/api/v1/status/UUID
# {"application_id": "UUID", "status": "COMPLETED", "updated_at": "..."}

curl -s -H "X-API-Key: dev-key-read-only" \
  http://localhost:8000/api/v1/decision/UUID
# {
#   "decision": "APPROVE",
#   "confidence": 0.7,
#   "data_reliability": 1.0,
#   "risk_score": 64.1,
#   "rule_version": "RULE_SET_V1",
#   "scoring_strategy": "heuristic"
# }

curl -s -H "X-API-Key: dev-key-read-only" \
  http://localhost:8000/api/v1/explanation/UUID
# {
#   "decision": "APPROVE",
#   "summary": "Decision APPROVE was produced from verified data sources with confidence 0.70.",
#   "timeline": [
#     {"step": "PROCESSING_STARTED", "status": "PROCESSING"},
#     {"step": "CREDIT_BUREAU_FETCH", "status": "SUCCESS"},
#     {"step": "GST_VERIFIER_FETCH", "status": "SUCCESS"},
#     {"step": "BANK_ANALYZER_FETCH", "status": "SUCCESS"},
#     {"step": "DECISION_CALCULATION", "status": "APPROVE"}
#   ],
#   "rule_version": "RULE_SET_V1"
# }
```

### 6.2 Failure Scenarios

#### 6.2.1 Credit Bureau Timeout

Given `failure_flags: {"credit_bureau": "TIMEOUT", ...}`:

1. `CreditBureauService.call()` receives `fail_mode=TIMEOUT` but the mock responds with HTTP 408 (or times out).
2. The service retries up to `MAX_RETRIES` times with exponential backoff.
3. After exhausting retries, `classify_response()` returns `FailureType.TIMEOUT`.
4. The fallback credit score `600` is used in the scoring function.
5. `compute_data_reliability()` applies a 0.30 penalty for `TIMEOUT`.
6. `compute_decision_confidence()` uses the reduced data reliability.
7. If confidence falls below 0.6, the decision becomes `NEEDS_REVIEW`.
8. Audit entry records `CREDIT_BUREAU_FETCH` with `error_type=TIMEOUT` and `fallback_used=True`.

#### 6.2.2 All External Services Down

Given `failure_flags: {"credit_bureau": "SERVICE_DOWN", "bank_analyzer": "FORMAT_ERROR", "gst_verifier": "NO_RECORD"}`:

1. All three services fail. The worker applies conservative fallbacks.
2. `compute_data_reliability()` applies all three penalties (0.30 + 0.30 + 0.10 = 0.70 penalty, capped at `max_data_quality_penalty = 15.0`).
3. Data reliability drops to ~0.30.
4. Confidence drops further due to the boundary distance factor.
5. The application is routed to `MANUAL_REVIEW` because confidence < 0.6.
6. The audit timeline records each failing step.

#### 6.2.3 GST PAN Mismatch

Given `failure_flags: {"gst_verifier": "PAN_MISMATCH"}`:

1. `GstVerifierService` returns `FailureType.PAN_MISMATCH`.
2. GST compliance is explicitly `False` (not `None`).
3. `evaluate()` applies the GST gate: `effective_risk_score = min(risk_score, approve_moderate_threshold - 1.0)`. This caps the effective risk score at 54.0.
4. If the heuristic risk score was 64.1 (approvable), it gets capped to 54.0, which drops below the "moderate approval" threshold.
5. The application is routed to `NEEDS_REVIEW` unless another decline rule applies (e.g., high DTI).
6. The explanation notes: "Data quality issues recorded: GST Verifier: PAN_MISMATCH."

### 6.3 Real-World Examples of System Behavior

All examples are from the integration tests and the README smoke tests. Each is reproducible with the same input.

- **Idempotent replay**: Submit the same payload with the same `Idempotency-Key` header twice. The second request returns HTTP 200 with the same application ID. No duplicate is created. The idempotency record in PostgreSQL guarantees this.
- **Payload mismatch**: Submit the same `Idempotency-Key` with a different `monthly_income`. The second request returns HTTP 409 Conflict. The payload hash comparison catches this.
- **External data reuse**: Submit an application, let it start processing, then kill the worker. Restart the worker. The application transitions from PROCESSING back to PROCESSING (the claim statement includes stale processing recovery). The external data fetch reuses the already-persisted `ExternalData` rows instead of calling the mocks again.
- **Stuck processing recovery**: Submit an application. While it's PROCESSING, update its `updated_at` to 10 minutes ago (simulating a worker crash). The claim statement picks it up because `status = 'PROCESSING' AND updated_at < stale_before`.
- **ML forced fallback**: Submit with `failure_flags.ml_model = "FORCE_LOW_CONFIDENCE"`. The ML scorer returns a fake low confidence (0.4). The worker falls back to the heuristic scorer. The decision output shows `scoring_strategy: "heuristic"` but `model_version: "XGB_V1"`.

---

## 7. Data Layer & State Management

### 7.1 Database Schema / Structure

The database has six tables, defined via SQLAlchemy models and managed by Alembic migrations.

```sql
-- loan_applications: the core entity
CREATE TABLE loan_applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idempotency_key VARCHAR(255) NOT NULL,
    pan_hash VARCHAR(64) NOT NULL,                       -- salted SHA-256, not raw PAN
    encrypted_user_data BYTEA NOT NULL,           -- AES-256-GCM ciphertext
    encryption_nonce BYTEA NOT NULL,               -- 12-byte nonce
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    decision VARCHAR(30),                        -- APPROVE, DECLINE, NEEDS_REVIEW
    confidence NUMERIC(3,2),                     -- 0.00 to 1.00
    failure_flags JSONB,                         -- for testing
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_loan_status ON loan_applications(status);
CREATE INDEX idx_loan_idempotency ON loan_applications(idempotency_key);
CREATE INDEX idx_loan_pan_hash ON loan_applications(pan_hash);

-- idempotency_records: ensures request/response replay
CREATE TABLE idempotency_records (
    key VARCHAR(255) PRIMARY KEY,
    application_id UUID REFERENCES loan_applications(id),
    response JSONB NOT NULL,                     -- includes _request_hash
    created_at TIMESTAMPTZ DEFAULT now()
);

-- outbox: transactional outbox pattern
CREATE TABLE outbox (
    id BIGSERIAL PRIMARY KEY,
    task_name VARCHAR(255) NOT NULL,
    task_args JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ,
    error_message TEXT
);
CREATE INDEX idx_outbox_status_created ON outbox(status, created_at);

-- external_data: cached provider responses (for reuse on retry)
CREATE TABLE external_data (
    id BIGSERIAL PRIMARY KEY,
    application_id UUID REFERENCES loan_applications(id),
    source_type VARCHAR(30) NOT NULL,             -- CREDIT_BUREAU, BANK_ANALYZER, GST_VERIFIER
    request_params JSONB,
    response_data JSONB,
    failure_type VARCHAR(30),
    idempotency_key VARCHAR(255) NOT NULL,
    fetched_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(application_id, source_type)
);
CREATE INDEX idx_external_data_app ON external_data(application_id, source_type);

-- audit_logs: append-only compliance trail
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    application_id UUID REFERENCES loan_applications(id),
    step VARCHAR(100) NOT NULL,                 -- PROCESSING_STARTED, CREDIT_BUREAU_FETCH, DECISION_CALCULATION, etc.
    input_snapshot JSONB,
    output_snapshot JSONB,
    error_type VARCHAR(50),                  -- FailureType value
    fallback_used BOOLEAN DEFAULT false,
    fallback_reason TEXT,
    rule_version VARCHAR(20),
    actor VARCHAR(30) NOT NULL DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_audit_app_step ON audit_logs(application_id, created_at);
```

### 7.2 Data Flow and Storage Logic

- **PII storage**: The entire `user_data` JSON object is encrypted as a single blob. Individual fields are not encrypted separately. This is simpler and safer — the ciphertext is opaque and can only be decrypted with the correct key and nonce.
- **PAN handling**: Raw PAN is never stored in any table. Only the salted SHA-256 hash is stored in `loan_applications.pan_hash`. The hash is used for idempotency and duplicate detection but cannot be reversed to recover the PAN.
- **External data storage**: `ExternalData.response_data` stores the full provider response. This is used to avoid redundant external calls on worker retry. The data is redacted before being written to audit logs (`_redact_user_data()` replaces sensitive fields with `***REDACTED***`).
- **Idempotency storage**: `IdempotencyRecord.response` stores the full API response plus a private `_request_hash` field. The hash is used to verify that a replay uses the same payload. The response is stored in JSONB to preserve the exact response format.
- **Outbox storage**: The outbox table is the transactional outbox. When the API writes the application, it also writes the outbox message in the same transaction. This guarantees that the application is not "committed" until the processing intent is persisted.

### 7.3 Caching, Indexing, Optimization Techniques

| Technique | Where | Why |
| --- | --- | --- |
| Redis idempotency cache | API | Fast path for idempotency checks. TTL = 86400s. Falls back to PostgreSQL on miss or failure. |
| Redis circuit breaker | External service clients | Stores state (CLOSED/OPEN/HALF_OPEN), failure count, last failure timestamp, probe lock. All keys have TTLs to auto-expire stale state. |
| PostgreSQL indexes | loan_applications(status), idempotency_key, pan_hash; audit_logs(application_id, created_at); outbox(status, created_at); external_data(application_id, source_type) | Indexes support the most frequent queries. The compound indexes match the query patterns in the worker. |
| JSONB columns | failure_flags, idempotency_response, task_args, input_snapshot, output_snapshot, response_data, request_params | Flexibility for structured data without a fixed schema. JSONB is indexed as a column, not as a full-text index. |
| `with_for_update()` | Worker updates loan_application at the end | Row-level locking prevents race conditions when storing results. |

---

## 8. Key Design Patterns Used

### 8.1 Identify Patterns

| Pattern | Location | Why Used |
| --- | --- | --- |
| **Transactional Outbox** | `api/routes/applications.py` writes to `loan_applications` + `outbox` + `idempotency_records` in a single transaction | Guarantees that the processing intent is persisted atomically with the application. No "committed but unprocessed" state is possible. |
| **Idempotency** | Redis fast path + PostgreSQL fallback with payload hash verification in `api/routes/applications.py` | Eliminates duplicate processing from network retries, client retries, worker restarts. The payload hash prevents "same key, different payload" replay attacks. |
| **Circuit Breaker** | `services/base.py` with Redis-backed state (CLOSED/OPEN/HALF_OPEN) | Prevents cascading failures when an external service is down. When the circuit opens, subsequent calls fail fast. |
| **Retry with Exponential Backoff + Deterministic Jitter** | `services/base.py._retry_delay()` | Prevents thundering herd on transient failures. The jitter is deterministic (`(attempt + 1) * 137 % 500 / 1000`) so retries are reproducible. |
| **External Data Reuse** | `worker/tasks/process_application.py._fetch_or_reuse_external_data()` | On worker retry, the external data has already been fetched and stored in `ExternalData`. The worker reuses it instead of calling the provider again. |
| **Atomic Claiming** | `worker/tasks/process_application.py._claim_application()` | `UPDATE ... WHERE status=PENDING OR (status=PROCESSING AND updated_at < stale_before)` with `RETURNING`. Prevents duplicate processing by multiple workers. |
| **Separation of Concerns: Risk Score, Data Reliability, Calibrated Confidence** | `engine/scoring.py`, `engine/confidence.py`, `engine/decision.py` | Three separate concepts: how risky the applicant is (risk score), how trustworthy the data is (data reliability), and how confident the decision is (confidence). This is required by the README invariants. |
| **Immutable Rule Sets** | `engine/rule_sets.py` frozen dataclasses | Changing weights or thresholds creates a new RuleSet version. The old version is preserved in audit logs. No in-place mutation of rules. |
| **Append-Only Audit** | `models/audit_log.py` + PostgreSQL trigger (`migrations/versions/20260429_0005_audit_protection.py`) | UPDATE/DELETE on audit_logs raises an exception. This is a regulatory requirement. |
| **Explanation from Audit Trail** | `engine/explanation_builder.py` builds from audit logs | The explanation must describe what actually happened at decision time, not recompute from current business logic. |
| **Fallback Chain** | `services/base.py`, `engine/scoring.py`, `engine/decision.py` | External service failure → fallback data + reduced data reliability. ML failure → fallback to heuristic. Confidence too low → manual review. Every fallback is audited. |
| **Deterministic Mocks** | `mock_apis/` | Identical input → identical output. Request IDs are derived from input, not random. This makes integration tests reproducible. |
| **A/B Experiment Assignment** | `ml/governance/ab_test.py` | Deterministic arm assignment based on application_id. No random assignment. Each application lands in exactly one arm. |
| **SHAP Explanations** | `ml/explain/shap_explainer.py` | Model-agnostic per-prediction explanations. Required for the "human-readable explanations" requirement. |
| **Drift Detection** | `ml/governance/drift_detector.py` | KS-based feature drift vs. reference snapshot. Non-blocking alert written to audit. |

### 8.2 Where and Why They Are Used

The patterns above are not accidental — they are the direct result of the project's stated invariants in `CLAUDE.md`:

> "Never add `random()` to business logic."
> "Never add randomness to mock response bodies."
> "Never update or delete `audit_logs` rows."
> "Never skip the idempotency check before creating or processing an application."
> "Never bypass the Redis idempotency cache for API request replay."
> "Never let duplicate worker delivery create duplicate decisions."
> "Never throw or persist unclassified external-service failures."
> "Never silently drop fallback usage."
> "Never change rule weights or thresholds in place."

Each pattern enforces one or more of these invariants. The transactional outbox enforces the "processing intent is persisted" invariant. The idempotency pattern enforces the "duplicate prevention" invariant. The append-only audit enforces the "immutable audit trail" invariant. The circuit breaker enforces the "no cascading failures" invariant.

### 8.3 Benefits and Drawbacks in This Context

| Pattern | Benefit | Drawback |
| --- | --- | --- |
| Transactional Outbox | Exactly-once processing guarantee; no lost messages |额外的数据库写入; 需要轮询器或显式任务调度 |
| Idempotency | Eliminates duplicates; enables safe retry | 两层缓存管理 (Redis + PostgreSQL); 需要payload hash计算 |
| Circuit Breaker | Fail-fast; no cascading failures | 需要Redis; 状态需要TTL管理 |
| External Data Reuse | Reduces external calls on retry; faster retry | 需要持久化; 存储空间 |
| Atomic Claiming | Prevents duplicate processing | 需要 `with_for_update()` 或条件 UPDATE |
| Immutable Rules | Audit trail of rule versions; regulatory compliance | 需要新版本号; 代码库增长 |
| Append-Only Audit | Regulatory compliance; tamper-evident | 无法更正错误条目 (需要新条目更正) |

---

## 9. Performance & Scalability Considerations

### 9.1 Bottlenecks in the Current System

| Bottleneck | Location | Impact | Evidence |
| --- | --- | --- | --- |
| Single PostgreSQL writer | `loan_applications`, `audit_logs`, `outbox`, `external_data` writes | Above ~100 applications/second, the single writer becomes a bottleneck. Write amplification (multiple tables per application) amplifies this. | No read replicas in docker-compose.yml. No connection pooling optimization. |
| Redis idempotency cache | API `GET` on every request | If Redis is slow or unavailable, the API falls back to PostgreSQL (latency increase). | No Redis cluster. Single Redis instance. |
| External service calls | `services/base.py`, concurrent gather in worker | Three concurrent HTTP calls, each with up to 3 retries. External latency caps the task at 30s × 3 = 90s max. | `EXTERNAL_API_TIMEOUT_SECONDS = 30`, `MAX_RETRIES = 3`. |
| ML model loading | `engine/scoring.py`, eager preload at worker startup | If `ML_ENABLED=true`, the worker loads XGBoost + calibrator + SHAP at startup. Model files are ~50MB. Container startup is slow. | `preload_ml_scorer_from_env()` runs at worker startup. |
| Audit log writes | Per-step writes in `worker/tasks/process_application.py` | 5–10 audit entries per application. Synchronous writes. | No batched audit writes. |

### 9.2 How It Scales (or Doesn't)

| Scaling Dimension | Current Capacity | What Would Be Needed at Scale |
| --- | --- | --- |
|throughput (applications/second) | ~100/s (single writer) | PostgreSQL read replicas + PgBouncer connection pooling + sharding or move to a message queue (Kafka) |
| External provider calls | ~3 concurrent calls per application; 30s timeout each | Provider rate limits are the bottleneck. Add a provider proxy with rate limiting. |
| Audit log storage | Append-only; no cleanup | Archive strategy for audit logs (S3 + cold storage) after N days. The table will grow unbounded otherwise. |
| Redis cache | Single instance | Redis Cluster or Sentinel for high availability. |
| Worker concurrency | `--concurrency=2` in docker-compose.yml | Increase Celery worker concurrency, add more worker containers. |

### 9.3 Suggestions for Improvement

1. **Read replicas**: Add one or more read replicas for `/status`, `/decision`, `/explanation` reads. The writes are the bottleneck, not the reads.
2. **Connection pooling**: Use `aiodsa` or PgBouncer to manage PostgreSQL connections. The API opens a new async session per request.
3. **Batched audit writes**: Combine multiple audit entry writes into a single `executemany()` call. Currently, each entry is a separate `session.add()`.
4. **Audit log archiving**: After N days, archive audit logs to S3 (or object storage) and delete from the table. The current design has no retention policy.
5. **Async task dispatch**: Replace the transactional outbox poller with Celery's direct task dispatch (`apply_async`). The outbox pattern is currently used but could be simplified.
6. **Provider timeout reduction**: Decrease `EXTERNAL_API_TIMEOUT_SECONDS` if the providers are known to respond faster. The 30s default is conservative.
7. **Celery result backend**: Use Redis for the result backend instead of the database (currently it is the database via `DATABASE_URL`). This would reduce database load.
8. **ML model preloading decision**: Load the ML model on-demand the first time it's needed, then cache it in memory. This speeds up worker startup.

---

## 10. Weaknesses & Limitations

### 10.1 Design Flaws or Risks

| Issue | Severity | Description |
| --- | --- | --- |
| No rate limiting on API | High | A malicious client can submit unlimited applications. DDoS by application submission. |
| API key is static | Medium | No key rotation. No expired keys. No scoped keys (beyond read/write scope). Real deployment requires OAuth2/OIDC. |
| Encryption key at rest in env | Medium | The `PII_ENCRYPTION_KEY` is passed as an environment variable. In containers, this is visible in process list. Real deployment needs a secrets manager (Vault, AWS Secrets Manager). |
| No request/response signing | Low | The API response is not signed. A man-in-the-middle could modify the decision. TLS/mTLS is needed. |
| Hardcoded confidence threshold (0.6) | Medium | One threshold for all applicants. Real credit decisions are risk-tiered. The threshold should be per-rule-set-version or per-risk-tier. |
| No data retention policy | High | Audit logs grow unbounded. No cleanup, no archiving. At scale, this fills the disk. |
| External data stored in plain JSONB | Medium | `ExternalData.response_data` is stored unencrypted. It's redacted before audit, but it's stored in the raw DB. |
| No request logging (audit of who did what) | Low | The audit log tracks system steps, but not which API key or user made the request. The `actor` field defaults to "system". |
| Celery result backend uses database | Medium | Each task completion writes to the database result backend. This adds write load. Redis would be faster. |

### 10.2 Technical Debt Areas

| Area | Why It's Debt | Refactoring Suggestion |
| --- | --- | --- |
| Hardcoded failure penalties | `engine/confidence.py` has hardcoded penalty values (0.30, 0.20, etc.). These are not configurable. | Move to a `RuleSet`-like configuration. |
| Confidence threshold not tiered | Single threshold for all decision types. Should be per decision type (APPROVE vs DECLINE). | Add per-decision-type thresholds to RuleSet. |
| Retry delay jitter uses a magic number | `(attempt + 1) * 137 % 500` is arbitrary. | Make the multiplier configurable. |
| No dead letter queue | Failed tasks that raise exceptions are marked MANUAL_REVIEW but are not tracked anywhere. | Add a dead letter table or retry logic with max attempts. |
| Outbox poller is not used | The transactional outbox pattern is implemented but Celery is dispatching tasks directly (via the API writing to outbox triggers a Celery task). The README mentions an outbox poller but it's not in the current flow. | Either remove the outbox poller code or use it. Currently it's dead code. |
| No version migration for decision engine | When a new `RuleSet` version is introduced, existing decisions in the database still have the old `rule_version` in the output. This is correct (they were evaluated with the old rules), but there's no migration to add the old rules to a historical lookup. | Add a `rule_version_history` table that stores the full RuleSet as JSON at decision time. |

### 10.3 What Would Break Under Scale or Edge Cases

- **Very long DTI**: DTI > 1.0 (existing EMIs > monthly income). The code handles this: `dti_component = max(0.0, 1 - dti)`. If dti is 1.5, the component is negative, clamped to 0.0. The application is declined by the DTI rule.
- **Very high income**: Monthly income of 10,000,000. DTI becomes very small. Risk score is high. This is correctly handled.
- **Missing all external data**: All three providers fail. Data reliability drops to ~0.30. Confidence drops below 0.6. Application routes to MANUAL_REVIEW. Audit trail records each failure.
- **Duplicate PAN**: The same PAN with different names. `pan_hash` is the same, but `encrypted_user_data` is different. Two applications with the same PAN hash are possible. This is a fraud detection gap.
- **Negative income or loan amount**: Pydantic validation handles `monthly_income > 0`, `loan_amount > 0`. Negative values are rejected at input validation.
- **Zero tenure**: Handled: `tenure_months` defaults to 12 if not provided.
- **Network partition between worker and database**: The worker catches database exceptions and marks the application MANUAL_REVIEW. The audit log records the error.
- **Redis unavailable**: If Redis is unavailable, the circuit breaker falls back to no-op (always CLOSED). Idempotency falls back to PostgreSQL. The API continues to work but slower.

---

## 11. How to Improve This System

### 11.1 Concrete, Actionable Improvements

1. **Add rate limiting**: Use `fastapi-limiter` or a reverse proxy (nginx) with rate limiting. Per-API-key limits.
   ```python
   from fastapi_limiter import Limiter
   limiter = Limiter(key_func=get_remote_address)
   @router.post("/apply-loan")
   @limiter.limit("10/minute")
   async def apply_loan(...): ...
   ```

2. **Add key rotation**: Store API keys in a database table with `created_at`, `expires_at`, `last_used_at`. Revoke old keys automatically.
   ```sql
   CREATE TABLE api_keys (
       key_hash VARCHAR(64) PRIMARY KEY,
       scope VARCHAR(20) NOT NULL,
       created_at TIMESTAMPTZ DEFAULT now(),
       expires_at TIMESTAMPTZ,
       last_used_at TIMESTAMPTZ
   );
   ```

3. **Add audit log archiving**: After 90 days, move audit logs to S3 and delete from the table.
   ```python
   # Archive script (run weekly as cron)
   archived = session.execute(
       select(AuditLog).where(AuditLog.created_at < now() - timedelta(days=90))
   ).all()
   # Write to S3 as JSONL
   # Delete from table
   session.execute(
       delete(AuditLog).where(AuditLog.id.in_([a.id for a in archived]))
   ```

4. **Add dead letter queue**: Track failed applications in a dedicated table.
   ```sql
   CREATE TABLE dead_letter_queue (
       id BIGSERIAL PRIMARY KEY,
       application_id UUID REFERENCES loan_applications(id),
       error_type VARCHAR(50),
       error_message TEXT,
       created_at TIMESTAMPTZ DEFAULT now()
   );
   ```

5. **Add per-risk-tier thresholds**: Make confidence thresholds configurable per rule version.
   ```python
   @dataclass(frozen=True)
   class RuleSet:
       # ... existing fields ...
       approve_confidence_threshold: float = 0.6
       decline_confidence_threshold: float = 0.4  # NEW
       manual_review_threshold: float = 0.6  # existing CONFIDENCE_THRESHOLD
   ```

6. **Add TLS/mTLS**: Terminate TLS at the load balancer. Service-to-service communication should be mTLS.
   ```yaml
   # docker-compose.yml (example)
   api:
     ports:
       - "8000:8000"
     # Add TLS config
   ```

7. **Add request signing**: Sign API responses with HMAC. Verify on read.
   ```python
   def sign_response(payload: dict, secret: str) -> str:
       return hmac.new(secret.encode(), json.dumps(payload).encode()).hexdigest()
   ```

8. **Migrate Celery result backend to Redis**:
   ```python
   # worker/celery_app.py
   celery_app = Celery('auditlend', broker=REDIS_URL, backend=REDIS_URL)
   ```

### 11.2 Better Architectural Alternatives

If the system were redesigned from scratch:

1. **Replace the transactional outbox with Kafka**: Kafka provides log persistence, ordering, and exactly-once semantics. The current PostgreSQL outbox works but caps throughput. Kafka would handle 10,000+ applications/second.
2. **Replace PostgreSQL with a time-series DB for audit**: TimescaleDB or QuestDB for audit logs. Append-optimized. Automatic data retention.
3. **Replace Celery with a custom async pipeline**: The Celery task is essentially an async pipeline. A custom pipeline using `asyncio` and a message queue (Kafka) would have lower latency.
4. **Add a decision versioning service**: Store every RuleSet version in a dedicated service. The decision engine queries the version at decision time. No code changes for new rules.
5. **Add a fraud detection service**: Current design doesn't detect fraud (same PAN with different names). Add a fraud detection step that checks for duplicate PAN hashes, velocity, and velocity across PAN hashes.

### 11.3 Refactoring Suggestions

| Current | Proposed | Why |
| --- | --- | --- |
| Hardcoded penalties in `engine/confidence.py` | Move to configuration or RuleSet | Configurable data quality penalties without code changes |
| Single confidence threshold | Per-decision-type thresholds | Risk-tiered thresholds |
| Static API keys | Database-backed keys with expiry | Key rotation, audit of key usage |
| No audit archiving | S3 archiving after 90 days | Disk space management |
| Celery result backend = PostgreSQL | Celery result backend = Redis | Reduce database write load |
| No dead letter tracking | Dead letter table | Failed application tracking |

---

## 12. Learning Notes (For a Developer)

### 12.1 Key Concepts to Study from This Project

1. **Deterministic business logic**: The entire decision engine (`engine/`) is pure. No randomness, no I/O, no network calls. This is a strong architectural principle. Study how `compute_risk_score()` and `evaluate()` are implemented. They take inputs and produce outputs with no side effects.
2. **Idempotency design**: The two-tier idempotency pattern (Redis + PostgreSQL) with payload hash verification is a robust pattern. Study `api/routes/applications.py` in detail.
3. **Append-only audit**: The combination of an ORM model (`AuditLog`), an audit write helper (`write_audit_entry()`), and a PostgreSQL trigger (`migrations/versions/20260429_0005_audit_protection.py`) is how immutability is enforced at the application and database layers.
4. **Transaction outbox pattern**: The API writes application + outbox + idempotency record in a single transaction. This is the standard pattern for ensuring that processing intent is persisted atomically.
5. **Circuit breaker pattern**: The Redis-backed circuit breaker in `services/base.py` is a good reference implementation. It uses three states (CLOSED/OPEN/HALF_OPEN), failure counting, and a half-open probe lock.
6. **Retry with deterministic jitter**: Exponential backoff with jitter that is reproducible. The formula `(attempt + 1) * 137 % 500 / 1000` is deterministic — it produces the same delay for the same attempt number every time.
7. **Separation of risk score, data reliability, and confidence**: These are three distinct concepts. Understanding the separation in `engine/confidence.py` and `engine/decision.py` is critical for understanding how the system handles degraded data quality.
8. **External data reuse**: The worker reuses persisted external data on retry. This pattern (check before fetch) is important for idempotent retries.
9. **SHAP explanations**: The ML platform includes SHAP explanations. Study `ml/explain/shap_explainer.py` to understand how per-prediction model explanations are generated.
10. **Drift detection**: The KS-based drift detection is non-blocking — it alerts but doesn't block the decision. This is a good pattern for production ML governance.

### 12.2 What Skills This Project Demonstrates

- **Python async**: FastAPI, httpx, asyncio.gather. The API is async-first, but the worker uses `asyncio.run()` to bridge to async from Celery.
- **SQLAlchemy 2.x async**: Async sessions, `select()`, `update()`, `with_for_update()`.
- **PostgreSQL**: JSONB, UUID, triggers, indexes, transactional semantics.
- **Redis**: Cache, idempotency, circuit breaker state, broker.
- **Celery**: Task queues, retry policies, result backend.
- **Cryptography**: AES-256-GCM encryption, SHA-256 hashing.
- **ML engineering**: XGBoost, SHAP, isotonic calibration, drift detection, model governance.
- **Observability**: Prometheus metrics, structured logging (structlog), health endpoints.
- **API design**: RESTful endpoints, idempotent POST, proper HTTP status codes (200, 201, 202, 401, 409), Problem Details error format.
- **Testing**: Unit, integration, chaos tests. The chaos tests (circuit breaker, idempotency under load, retry exhaustion) are especially educational.

### 12.3 How to Replicate or Build Something Similar

1. **Start with the data model**: Define the SQLAlchemy models first. The application, idempotency, outbox, external data, and audit log tables are the backbone.
2. **Implement idempotency first**: Redis + PostgreSQL with payload hash. This is the foundation of the entire system.
3. **Implement the transactional outbox**: Write application + outbox in a single transaction.
4. **Implement the decision engine as pure functions**: Start with `compute_risk_score()` and `evaluate()`. No I/O, no randomness.
5. **Add data reliability and confidence**: Separate these from risk score.
6. **Add external service clients**: Wrap httpx with retry, backoff, and circuit breaker.
7. **Add audit writes**: After every significant step, write an audit entry.
8. **Add explanation builder**: Read audit entries and construct an explanation.
9. **Add ML (optional)**: XGBoost model, SHAP explanations, drift detection.

The critical principle is **determinism**: every function in `engine/` must produce the same output for the same input. This makes the system testable, reproducible, and auditable.

---

*This study guide was generated from a deep analysis of the AuditLend Intelligence Core codebase on 2026-05-03. The guide reflects the codebase at that point in time. The project is in active development; some implementation details may change over time.*