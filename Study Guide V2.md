# AuditLend Intelligence Core (ALICe) — Comprehensive Study Guide V2

> **About this document:** This is the second edition of the study guide, built as a follow-on to `STUDY_GUIDE.md` (V1). It reuses the same pedagogical structure — first principles → architecture → implementation → weaknesses → learning notes — but reflects the **current architecture** of the codebase, which has grown well beyond the core pipeline documented in V1. New material covers the ML platform (causal inference, survival analysis, uplift, portfolio analytics, optimization), the expanded API surface (batch, documents, monitoring), security hardening (OAuth2/OIDC, rate limiting, security headers), Vault integration, LLM narratives, and policy RAG.
>
> V1 remains the canonical deep-dive on the **core decision pipeline** (idempotency, outbox, circuit breaker, audit). V2 assumes that material is familiar and emphasizes what changed, what is new, and the full current system. An appendix (§13) summarizes the delta between V1 and V2.

---

## 1. Project Overview

### 1.1 What the Project Does (Practical Terms)

AuditLend Intelligence Core is a loan application processing system that:

1. Receives credit applications over a FastAPI endpoint (`POST /api/v1/apply-loan`).
2. Fetches data from three deterministic external providers (credit bureau, bank analyzer, GST verifier).
3. Computes a risk score using either a deterministic heuristic scorecard (**RULE_SET_V1**) or a calibrated XGBoost model (**RULE_SET_V2**, XGB_V1) that falls back to the heuristic on low confidence.
4. Applies immutable rule sets to approve, decline, or route to manual review.
5. Produces a human-readable explanation derived **entirely from the audit trail** — never recomputed from current logic.
6. Records every step in an append-only audit log that cannot be updated or deleted (enforced by a PostgreSQL trigger).

Beyond the core pipeline, the current system is a **full ML lifecycle platform**:

- **Calibrated ML scoring**: XGB_V1 (AUC-ROC 0.9757, ECE 0.0036) with SHAP per-prediction explanations.
- **Governance**: file-backed model registry, KS-test drift detection, deterministic A/B experiment assignment, champion/challenger promotion logic.
- **Causal inference**: propensity score matching (PSM), synthetic control, and a statistical A/B framework.
- **Survival analysis**: Kaplan–Meier and Cox proportional hazards for time-to-default.
- **Uplift modeling**: two-model UpliftXGB for individual treatment effects.
- **Portfolio analytics**: risk aggregation, stress testing, concentration (HHI), Markdown/HTML reporting, and an `auditlend-portfolio` CLI.
- **Performance engineering**: ONNX export, async model loader, deterministic prediction cache.
- **Explainability**: SHAP + optional LLM narrative generation (with PII stripping and deterministic template fallback) + policy RAG grounding against `docs/policy_corpus/`.
- **Production hardening**: OAuth2/OIDC + API-key composite auth, in-process rate limiting, security headers, request size limits, optional HashiCorp Vault for PII key management, multi-modal document parsing (PDF/OCR) for bank statements, salary slips, and GST filings.

The headline business result (from `docs/BUSINESS_IMPACT.md`): on a held-out 2018 test set of 49,230 loans, the ML scorer increased approvals by 0.6pp **and** reduced the approved-portfolio default rate from 15.06% (heuristic) to 2.35% — an 84% reduction — producing a simulated **$68.3M profit delta** over the heuristic baseline.

### 1.2 Core Problem It Solves

Lending decisions must be:

- **Deterministic**: identical inputs produce identical outputs. No `random()` in business logic anywhere. Even the retry jitter and A/B arm assignment are deterministic.
- **Idempotent**: network retries, worker restarts, and duplicate Celery deliveries must not create duplicate decisions. Redis fast path + PostgreSQL durable fallback + content-addressed Celery task IDs.
- **Auditable**: every input, external call, intermediate computation, fallback, and ML artifact version is captured in an immutable trail.
- **Resilient**: typed `FailureType` classification, retry with backoff, Redis-backed circuit breakers, conservative fallbacks with confidence penalties.
- **Explicable**: explanations describe what happened at decision time (from audit entries), and are now optionally enriched by LLM narratives grounded in the policy corpus.
- **Governable (new in V2)**: models are registered, drift is monitored, experiments are assigned deterministically, and ML performance is measured against business outcomes (profit, default rate).

### 1.3 Key Features and Capabilities

| Feature | Implementation |
| --- | --- |
| Loan application intake | FastAPI `POST /api/v1/apply-loan` — returns immediately after transactional write |
| Asynchronous processing | Celery worker + **outbox poller** (polls `outbox` with `FOR UPDATE SKIP_LOCKED`, dispatches with deterministic content-addressed task IDs) |
| Idempotency | Redis fast path + PostgreSQL durable fallback, SHA-256 payload-hash verification, 409 on hash mismatch |
| PII protection | AES-256-GCM encryption; salted SHA-256 PAN hash (raw PAN never stored); optional HashiCorp Vault for key/salt |
| Append-only audit | SQLAlchemy model + PostgreSQL trigger blocking UPDATE/DELETE on `audit_logs` |
| Heuristic risk scoring | Weighted 0–100 score in `engine/scoring.py` (credit, income stability, DTI, GST, data-quality penalty) |
| ML risk scoring | XGB_V1 (XGBoost, isotonic-calibrated, ECE 0.0036), SHAP top-8 contributions, KS drift detection per decision |
| Scoring fallback | RULE_SET_V2 (ML-assisted) falls back to RULE_SET_V1 heuristic on low model confidence — always audited |
| Data reliability | Per-`FailureType` penalties in `engine/confidence.py`, separate from risk score |
| Calibrated confidence | `data_reliability × boundary_distance_factor`; override to `NEEDS_REVIEW` below threshold (default 0.6) |
| Decision rules | Priority-ordered matching in `engine/rules.py`; GST non-compliance gates automatic approval |
| External service resilience | Retry + exponential backoff + deterministic jitter; Redis circuit breaker (CLOSED/OPEN/HALF_OPEN); typed `FailureType` |
| Explanations | Audit-trail reconstruction (`engine/explanation_builder.py`) + optional LLM narrative + policy RAG citations |
| Multi-modal document parsing | `services/document_parser.py` — text/PDF/OCR (tesseract) parsing of bank statements, salary slips, GST filings |
| Batch prediction | `POST /api/v1/batch/predict` (scaffold) + `PredictionCache` LRU/TTL cache, ONNX export, async model loader |
| Monitoring | `POST /api/v1/monitoring/drift` (Evidently or KS fallback), drift reports, Prometheus metrics, Grafana dashboards |
| A/B experimentation | Deterministic sha256 bucket assignment (`ml/governance/ab_test.py`), statistical framework, champion/challenger verdicts |
| Causal inference | PSM (ATT), synthetic control, bootstrap CIs (seeded) |
| Survival analysis | Kaplan–Meier (Greenwood CI) + CoxPH (Breslow ties, hazard ratios, concordance) |
| Uplift modeling | Two-model UpliftXGB, Qini coefficient |
| Portfolio analytics | Risk buckets, percentiles, stress tests, HHI concentration, `auditlend-portfolio` CLI |
| Experiment tracking | Optional MLflow integration (disabled by default; registry remains source of truth) |
| Security | CompositeAuth (API key + OAuth2/OIDC JWT), token-bucket rate limiting, security headers, request size limits |

---

## 2. High-Level Architecture

### 2.1 Overall System Design

```
 Client (REST / CLI auditlend-portfolio)
   |
   | POST /api/v1/apply-loan          (X-API-Key or OAuth2/OIDC bearer, Idempotency-Key)
   v
 FastAPI API (port 8000)  ── rate limiter, security headers, request size limits ──
   | \
   |  \-- PostgreSQL (ACID source of truth)
   |       - loan_applications (encrypted PII)
   |       - idempotency_records (request/response pairs)
   |       - outbox (transactional outbox)
   |       - external_data (cached provider responses)
   |       - audit_logs (append-only, trigger-protected)
   |
   \-- Redis (cache / broker)
        - idempotency cache (fast path)
        - Celery broker + result backend
        - circuit breaker state per external service
        |
        v
 Celery Worker
   |  - outbox_poller: polls outbox (FOR UPDATE SKIP_LOCKED),
   |    dispatches celery task with deterministic task_id f"{task_name}:{application_id}"
   |  - atomically claims applications (PENDING or stale PROCESSING)
   |  - reuses persisted external_data on redelivery
   |  - heuristic scoring (RULE_SET_V1) and/or ML scoring (RULE_SET_V2, XGB_V1)
   |  - SHAP explain → calibrate → drift check → A/B assignment
   |  - writes audit trail + Prometheus metrics
   |
   v (external calls with retry / circuit breaker / typed failure)
 Credit Bureau Mock (:8001)  Bank Analyzer Mock (:8002)  GST Verifier Mock (:8003)

 ML Layer (optional, degraded gracefully when deps missing):
   XGB_V1 model + isotonic calibrator + SHAP explainer
   ONNX export  |  Prediction cache  |  Async model loader
   Drift: KS-test (operational) + Evidently (optional dashboards)
   Governance: ModelRegistry (file-backed JSON) + MLflow (optional) + A/B framework
   Analytics: causal (PSM/SC), survival (KM/Cox), uplift, portfolio risk, CLI reports
```

This is a **layered async architecture**: the API layer is synchronous-to-async (single transactional write, immediate response), the worker layer is fully async (Celery task wrapping `asyncio`), and all external calls are concurrent `httpx.AsyncClient` requests via `asyncio.gather`. PostgreSQL is the source of truth; Redis is cache and broker only. Everything ML is deterministic or explicitly governed (see §2.4 for the one intentional exception).

### 2.2 Major Components and How They Interact

| Component | Responsibility | Key Files |
| --- | --- | --- |
| `api/` | HTTP handling, composite auth, rate limiting, security headers, idempotency, validation | `api/main.py`, `api/auth.py`, `api/routes/*`, `api/schemas/*` |
| `worker/` | Celery app, outbox poller, atomic claiming, task execution | `worker/celery_app.py`, `worker/outbox_poller.py`, `worker/tasks/process_application.py` |
| `engine/` | Pure scoring, decision rules, confidence, explanation building | `engine/scoring.py`, `engine/rules.py`, `engine/rule_sets.py`, `engine/confidence.py`, `engine/decision.py`, `engine/explanation_builder.py` |
| `services/` | Provider clients (retry/circuit breaker), crypto, Vault, document parsing, audit, metrics, logging | `services/base.py`, `services/crypto.py`, `services/vault.py`, `services/document_parser.py`, `services/audit.py`, `services/metrics.py` |
| `models/` | SQLAlchemy ORM models (5 tables) | `models/application.py`, `models/audit_log.py`, `models/idempotency.py`, `models/outbox.py`, `models/external_data.py` |
| `mock_apis/` | Deterministic provider doubles | `mock_apis/credit_bureau.py`, `bank_analyzer.py`, `gst_verifier.py` |
| `ml/` | Full ML lifecycle: data, models, explain, governance, causal, optimize, monitoring, portfolio, tracking | see §5.1 |
| `cli/` | `auditlend-portfolio` portfolio CLI | `cli/portfolio.py` |
| `docs/` | Architecture, business impact, calibration, A/B, survival, policy corpus | `docs/ARCHITECTURE.md`, `docs/BUSINESS_IMPACT.md`, `docs/CALIBRATION.md`, `docs/AB_TESTING.md`, `docs/SURVIVAL_ANALYSIS.md`, `docs/policy_corpus/*` |
| `research/` | EDA, WOE/IV, causal, survival notebooks | `research/*.ipynb`, `research/woe_iv_report.md` |

### 2.3 Data Flow Across the System

1. **Client submits** `POST /api/v1/apply-loan` (JSON `user_data` + optional `failure_flags` + `Idempotency-Key` header; `user_data.bank_statement` may carry client-supplied transaction rows).
2. **Auth & limits**: CompositeAuth checks API key (dev keys) or OAuth2/OIDC bearer JWT; token-bucket rate limiter; security headers applied; request size capped.
3. **Idempotency check**: Redis fast path → PostgreSQL durable fallback; hash mismatch → 409 Conflict.
4. **PII handling**: `PIIService.encrypt(user_data)` → AES-256-GCM ciphertext + nonce (key from env or Vault); `hash_pan(pan)` → salted SHA-256 stored in `pan_hash`.
5. **Transactional write**: `loan_applications` (status PENDING) + `outbox` (task_name, task_args) + `idempotency_records` committed in one transaction; response cached to Redis; HTTP 201.
6. **Outbox poller** (`worker/outbox_poller.py`) selects PENDING/FAILED rows (`FOR UPDATE SKIP_LOCKED`) and dispatches `process_application` via `send_task(task_id=f"{task_name}:{application_id}")` — the deterministic task ID prevents duplicate deliveries even if polling races.
7. **Worker claims** atomically: `UPDATE ... SET status='PROCESSING' WHERE id=:id AND (status='PENDING' OR (status='PROCESSING' AND updated_at < now() - 300s))`. Terminal/locked applications return stored results instead.
8. **External fetch** (concurrent): each provider call goes through `BaseExternalService.call()` — circuit breaker check, retries with exponential backoff + deterministic jitter, typed classification. `_fetch_or_reuse_external_data` reuses persisted `external_data` rows on redelivery.
9. **Decision computation** (`engine/decision.py`): heuristic score → (if `ML_ENABLED`) ML score with SHAP, calibration, drift detection → rule evaluation (RULE_SET_V2 if ML used, else V1) → data reliability → calibrated confidence → confidence override to `NEEDS_REVIEW` → deterministic A/B arm assignment.
10. **Store + audit**: application status/decision/confidence updated; `ExternalData` rows persisted; audit entries written per step (`PROCESSING_STARTED`, `{SOURCE}_FETCH`, `ML_SCORING`, `DRIFT_DETECTED`, `DECISION_CALCULATION`, `MANUAL_REVIEW_OVERRIDE`); Prometheus metrics (A/B assignments, drift alerts, task duration) incremented.
11. **Client reads**: `/status/{id}`, `/decision/{id}`, `/explanation/{id}` (audit-trail reconstruction), optionally `/monitoring/*`, `/batch/*`, `/apply-loan/documents`.

### 2.4 Determinism and the One Intentional Exception

The V1 invariants all still hold: no `random()` in business logic; deterministic mock bodies; deterministic retry jitter (`(attempt+1) * 137 % 500 / 1000`); deterministic A/B assignment (`sha256(application_id) % 10000`); seeded ML training (`random_state=42`); seeded bootstrap (`random.Random(42)`); no randomness in PSM, synthetic control, KS, CoxPH, KM, Qini, or portfolio math.

The **one intentional exception** is LLM narrative generation (`ml/explain/llm_narrative.py`, temperature 0.3). It is governed rather than deterministic: PII is stripped before any external call, refusal responses are detected, output is schema-constrained, and any failure falls back to a deterministic template flagged with `fallback_used=True`. Policy RAG citations (`ml/explain/policy_rag.py`) are deterministic for a fixed corpus and cite `source §section`, so LLM output remains auditable even when the prose is not byte-reproducible.

---

## 3. Why This Architecture?

### 3.1 Why This Architecture Was Chosen Over Alternatives

The V1 rationale is unchanged for the core pipeline (async worker isolation, transactional outbox, Redis+Postgres idempotency, append-only audit, ML in the worker with heuristic fallback). Additions in V2:

| Alternative | Why Rejected | What V2 Chose |
| --- | --- | --- |
| Single API-key auth | No rotation, no scopes, no standards | **CompositeAuth**: `X-API-Key` (dev) **or** OAuth2/OIDC bearer JWT (`python-jose`), enabling enterprise SSO |
| No rate limiting | Malicious clients can flood the queue | In-process token-bucket rate limiter (`api/auth.py`), per-client configurable limits |
| Secrets only in env vars | Visible in process lists, no rotation story | Optional HashiCorp Vault (`services/vault.py`) for PII key/salt with env fallback |
| Unstructured provider data | Hard to audit | All provider results stored as typed `ServiceResult` + JSONB `external_data` snapshots |
| ML scoring inline in API | Blocks request thread; cold model loads | ML in the worker; `AsyncModelLoader` (`ml/optimize/async_loader.py`) loads the pickled model + SHAP explainer on a background thread; ONNX export for lower-latency inference; `PredictionCache` for repeat requests |
| Random A/B assignment | Not reproducible, not auditable | Deterministic sha256 bucketing; assignments persisted in audit + metrics |
| Unversioned models | Cannot roll back or explain past decisions | File-backed `ModelRegistry` (JSON) + optional MLflow; every decision records `model_version` and rule version |
| Static explanations | Recompute risks drift from decision-time truth | Explanations from audit trail; optionally enriched by LLM narrative grounded in policy corpus |

### 3.2 Trade-Offs

| Trade-off | Impact | Mitigation |
| --- | --- | --- |
| Async processing adds latency | No decision in the POST response; clients poll | Documented ~1–2s happy-path latency; immediate 201 response is a UX feature |
| Two-tier idempotency (Redis + Postgres) | Stale Redis after Postgres write | Payload hash always verified; Postgres authoritative |
| Optional ML deps | Feature availability varies by environment | Every optional dependency (MLflow, Evidently, ONNX, Vault, OIDC, OCR, LLM, ChromaDB) degrades gracefully with a capability flag (e.g., `EVIDENTLY_AVAILABLE`, `_ML_AVAILABLE`) |
| LLM narratives are non-deterministic | Same decision can produce different prose | PII stripping, refusal detection, JSON schema, deterministic template fallback with `fallback_used`, versioned prompt, policy citations |
| Outbox poller adds a hop | Slightly more machinery than direct `apply_async` | Poll is cheap (1s interval, `FOR UPDATE SKIP_LOCKED`); deterministic task IDs make it safe under concurrency |
| Model registry is file-backed JSON | No transactions, no CAS | Fine for single-writer governance workflows; MLflow offered as an optional alternative for teams needing a server |

### 3.3 When This Architecture Fails or Becomes Inefficient

- **Sub-second decisions**: the async pipeline floor of ~1–2s is unbeatable without synchronous inline scoring.
- **Very high throughput**: a single PostgreSQL writer bottlenecks above ~100 applications/s; batch ML scoring and prediction caching help batch workloads but not the transactional path.
- **Very long external calls**: >60s total (task timeout) forces `MANUAL_REVIEW`; 30s provider timeouts with 3 retries already bound the worst case.
- **Multi-region**: outbox assumes a single Postgres; the file-backed registry and in-process rate limiter are single-instance too.
- **LLM availability**: narrative generation depends on an external LLM; the fallback template keeps the API functional but loses personalization.
- **OCR quality**: `document_parser.py` confidence falls to 0.0 on unreadable scans — parse failures are warnings, not pipeline blockers (the parser is a standalone service).

---

## 4. Tech Stack Breakdown

### 4.1 Languages, Frameworks, Libraries, Databases

| Layer | Technology | Notes |
| --- | --- | --- |
| Language | Python 3.11+ | async-first |
| API framework | FastAPI | Pydantic v2 validation, automatic OpenAPI docs |
| Auth | `python-jose[cryptography]` | OAuth2/OIDC JWT bearer + API keys (CompositeAuth) |
| Task queue | Celery | Redis broker + result backend |
| Database | PostgreSQL 16-alpine | ACID, JSONB, UUID, triggers |
| Cache / broker | Redis 7-alpine | idempotency fast path, circuit breaker, Celery |
| ORM | SQLAlchemy 2.x async (asyncpg) | |
| HTTP client | httpx | async concurrent external calls |
| Encryption | pyca/cryptography (AESGCM) | AES-256-GCM PII; SHA-256 salted PAN hash |
| Secrets | hvac (optional) | HashiCorp Vault KV v2 with env fallback |
| Document parsing | pdfplumber/PyPDF2, pytesseract OCR, Pillow | multi-modal, regex/keyword extraction |
| ML | XGBoost, scikit-learn, shap | XGB_V1, isotonic calibration, SHAP |
| Causal/survival | scipy, statsmodels-style hand-rolled solvers | PSM, synthetic control, CoxPH (Newton–Raphson), KM |
| Vector search | chromadb (optional) | policy RAG |
| LLM | litellm/openai (optional), Ollama local | narrative generation |
| Experiment tracking | mlflow (optional) | disabled by default |
| Drift | evidently (optional) + internal KS | dashboards + JSON |
| Metrics | prometheus_client | /metrics endpoint |
| Logging | structlog | structured JSON |
| Migrations | Alembic | 5 migrations |
| Infra | docker compose, Grafana | local/dev only |

### 4.2 Why Each Was Likely Chosen

Most rationales are unchanged from V1 (FastAPI for async+validation, Celery for task maturity, Postgres for ACID outbox + audit triggers, Redis for latency, XGBoost for tabular credit risk, SHAP for explainability, structlog for JSON logs, Alembic for versioned schema).

New additions:

| Technology | Rationale |
| --- | --- |
| **OAuth2/OIDC** | Standard enterprise identity; lets the same API serve both dev keys and SSO-bearer clients without a bespoke auth system |
| **python-jose** | Lightweight JWT verification; the CompositeAuth design means OIDC is optional and additive |
| **hvac (Vault)** | Centralized secret management with rotation; the env fallback keeps the system runnable without it |
| **pdfplumber + pytesseract** | Deterministic extraction of structured data from the documents borrowers actually have; regex/keyword parsing (no ML in the parser) preserves determinism |
| **scipy (hand-rolled CoxPH/KM)** | Survival math without heavyweight statsmodels; hand-rolled Newton–Raphson keeps the dependency surface small and the math auditable |
| **chromadb** | Local, deterministic vector search for policy grounding; TF-IDF fallback removes the hard dependency |
| **ONNX / onnxruntime** | Portable, faster inference for batch scoring; the deterministic export path means ONNX predictions equal pickle predictions |
| **Evidently** | Off-the-shelf drift dashboards; the KS-based fallback (`ml/governance/drift_detector.py`) is the deterministic, always-available path |
| **MLflow** | Optional experiment tracking; the file-backed registry is the source of truth regardless, so MLflow absence never changes decisions |

### 4.3 What Alternatives Could Have Been Used and Why They Weren't

| Alternative | Why It Could Have Been Used | Why It Wasn't |
| --- | --- | --- |
| Django REST framework | Built-in auth/admin | Over-engineered; FastAPI async + Pydantic fits the async worker model better |
| RabbitMQ instead of Redis broker | Battle-tested broker | Redis already needed for idempotency/circuit breakers; one less service |
| Kafka instead of Postgres outbox | Higher throughput | Volume doesn't justify it; Postgres outbox keeps exactly-once intent atomic with the application row |
| statsmodels/lifelines for survival | Faster to implement, more features | The hand-rolled CoxPH/KM are deterministic, dependency-light, and fully testable; lifelines adds heavy deps |
| A dedicated ML serving server (e.g., MLflow Model Serving / KServe) | Managed latency, batching | Overkill for this workload; `AsyncModelLoader` + `PredictionCache` + ONNX deliver the same benefits in-process |
| Postgres for prediction cache | Simpler stack | LRU+TTL in-memory cache is faster and its deterministic keying (`sha256(sorted features + model_version)`) prevents cross-version replays |
| Real LLM vendor SDK only | One integration path | Multi-provider (remote/local/fallback) keeps the system runnable offline and auditable |

---

## 5. Folder & Code Structure Deep Dive

### 5.1 Explain Each Major Folder/Module

```
/Users/purvansh/Desktop/Projects/AuditLend Intelligence Core (ALICe)/
├── api/
│   ├── main.py                 # FastAPI app: middleware (security headers, size limits), auth, rate limiter, routes
│   ├── auth.py                 # CompositeAuth (API key + OAuth2/OIDC JWT), RateLimiter (token bucket)
│   ├── dependencies.py         # async SQLAlchemy session dependency
│   ├── routes/
│   │   ├── applications.py     # POST /apply-loan, GET /status/{id} — idempotency, encryption, transactional outbox write
│   │   ├── decisions.py        # GET /decision/{id}
│   │   ├── explanations.py     # GET /explanation/{id} — audit-trail reconstruction
│   │   ├── batch.py            # POST /batch/predict, GET /batch/status (scaffold; see §6.3)
│   │   ├── documents.py        # POST /apply-loan/documents — parse documents, return features (not stored)
│   │   └── monitoring.py       # POST /monitoring/drift, GET /monitoring/reports, POST /monitoring/reports/generate
│   └── schemas/
│       ├── application.py      # ApplyLoanRequest/Response, UserData (incl. bank_statement), FailureFlags
│       ├── decision.py         # decision output models
│       └── explanation.py      # explanation output models
├── worker/
│   ├── celery_app.py           # Celery app; ML preload at startup; starts outbox poller on worker_ready
│   ├── outbox_poller.py        # polls outbox (FOR UPDATE SKIP_LOCKED), dispatches with deterministic task IDs
│   └── tasks/
│       └── process_application.py  # claim → fetch/reuse external data → decide (heuristic/ML) → store + audit
├── engine/                     # PURE: no I/O, no network, no DB, no randomness
│   ├── scoring.py              # compute_risk_score() heuristic; MLScorer (XGB_V1 + calibration + SHAP + drift)
│   ├── rule_sets.py            # immutable RuleSet dataclasses (RULE_SET_V1, RULE_SET_V2, ACTIVE_RULE_SET)
│   ├── rules.py                # evaluate() priority-ordered rules; GST gate
│   ├── confidence.py           # data reliability penalties; calibrated confidence (boundary distance factor)
│   ├── decision.py             # compute_decision() orchestration; RULE_SET_V2 when ML used
│   └── explanation_builder.py  # human-readable explanations from audit logs
├── services/
│   ├── __init__.py             # FailureType enum, ServiceResult dataclass
│   ├── base.py                 # BaseExternalService: retry/backoff/jitter, Redis circuit breaker, typed classification
│   ├── credit_bureau.py        # + STALE_DATA detection, credit-600 fallback
│   ├── bank_analyzer.py        # + PARTIAL_DATA, FORMAT_ERROR fallback
│   ├── gst_verifier.py         # + PAN_MISMATCH/NO_RECORD → gst_compliant=False fallback
│   ├── crypto.py               # PIIService: AES-256-GCM, salted PAN hash, insecure-key/salt rejection
│   ├── vault.py                # optional HashiCorp Vault: key/salt retrieval, env fallback, create_pii_service_from_vault_or_env
│   ├── document_parser.py      # text/PDF/OCR parsing of bank statements, salary slips, GST filings → DocumentFeatures
│   ├── audit.py                # write_audit_entry() append-only
│   ├── metrics.py              # Prometheus metrics (batch, drift, cache, MLflow, A/B, circuit breaker, tasks)
│   └── logging.py              # structlog config
├── models/
│   ├── application.py          # LoanApplication (encrypted PII, pan_hash, status, decision, confidence)
│   ├── audit_log.py            # AuditLog — APPEND ONLY (DB trigger)
│   ├── idempotency.py          # IdempotencyRecord
│   ├── outbox.py               # OutboxMessage (status, processed_at, error_message)
│   └── external_data.py        # ExternalData (cached provider responses, unique per application+source)
├── migrations/versions/        # 5 Alembic migrations (initial → encryption → constraints → outbox/external → audit trigger)
├── mock_apis/
│   ├── credit_bureau.py        # SUCCESS/TIMEOUT/STALE_DATA/SERVICE_DOWN
│   ├── bank_analyzer.py        # SUCCESS/PARTIAL_DATA/FORMAT_ERROR
│   ├── gst_verifier.py         # SUCCESS/PAN_MISMATCH/NO_RECORD
│   └── run_all.py
├── ml/
│   ├── data/                   # ingestion.py, features.py (38 features), splits.py — Lending Club 2007–2018
│   ├── models/                 # train.py, evaluate.py, calibrate.py; XGB_V1 artifacts + manifest.yaml; survival_coxph.py, survival_km.py, uplift_xgb.py
│   ├── explain/                # shap_explainer.py, llm_narrative.py, policy_rag.py, prompts/
│   ├── governance/             # model_registry.py (file-backed JSON), drift_detector.py (KS), ab_test.py (deterministic)
│   ├── benchmark/              # heuristic_vs_ml.py + reports
│   ├── causal/                 # ab_framework.py, champion_challenger.py, psm.py, synthetic_control.py
│   ├── optimize/               # async_loader.py, onnx_export.py, prediction_cache.py
│   ├── monitoring/             # drift_reporter.py (Evidently + KS fallback)
│   ├── portfolio/              # risk_aggregator.py, report_generator.py
│   └── experiment_tracking.py  # MLflow wrapper (disabled by default)
├── cli/portfolio.py            # auditlend-portfolio (summary / stress-test / html-report)
├── demo/                       # portfolio analysis demo
├── grafana/dashboards/         # drift/ops dashboards
├── monitoring/reports/         # generated drift/test-suite reports (stubs by default)
├── tests/                      # unit (437+), integration (needs Postgres+Redis), chaos
├── docs/                       # ARCHITECTURE, BUSINESS_IMPACT, CALIBRATION, AB_TESTING, SURVIVAL_ANALYSIS, policy_corpus/
├── research/                   # EDA, WOE/IV, survival, causal notebooks
├── docker-compose.yml          # postgres, redis, 3 mocks, api, worker, flower (+ optional ML)
└── README.md
```

### 5.2 Responsibility of Each Component

| Component | Responsibility |
| --- | --- |
| `api/main.py` | App factory; middleware (security headers: HSTS/CSP/X-Frame-Options; request size limits); mounts routes; `/metrics` and health endpoints |
| `api/auth.py` | `require_auth` dependency backed by CompositeAuth — checks dev API keys first, falls through to OAuth2/OIDC JWT verification when configured; `RateLimiter` token bucket (per client, configurable refill/capacity) |
| `api/routes/applications.py` | Idempotency (Redis→Postgres) with payload hash; PII encryption; single-transaction write of application + outbox + idempotency record; metrics |
| `api/routes/batch.py` | Batch predict/status endpoints; ML availability probe; `PredictionCache` stats; **scaffold — `_predict_batch` returns a constant 0.5 per row** |
| `api/routes/documents.py` | Content-type whitelist (PDF/PNG/JPEG/plain text), 10MB cap, calls `parse_document_bytes`, returns features + warnings; **document is not stored, not audited, not fed server-side into the pipeline** |
| `api/routes/monitoring.py` | Drift check endpoint (Evidently or KS fallback, logs `drift_check`), report listing/generation; **note: `GET /monitoring/reports` is unauthenticated** |
| `worker/outbox_poller.py` | Polls `outbox` PENDING/FAILED with `FOR UPDATE SKIP_LOCKED`; dispatches `process_application` with `task_id=f"{task_name}:{application_id}"` (content-addressed, duplicate-safe); 1s default interval |
| `worker/tasks/process_application.py` | Atomic claim (PENDING or stale PROCESSING >300s); concurrent external fetch with `external_data` reuse; decision computation; result storage under `FOR UPDATE`; full audit trail; A/B + drift + task metrics; 60s task timeout → `_mark_manual_review_after_system_error` |
| `engine/scoring.py` | Pure heuristic `compute_risk_score()`; `MLScorer` (XGB_V1 load, feature mapping, calibrated probability, SHAP, drift detection) |
| `engine/rule_sets.py` | Immutable frozen dataclasses; weights/thresholds never mutated in place — new versions only |
| `engine/confidence.py` | `compute_data_reliability()` (per-FailureType penalties, capped penalty), `compute_decision_confidence()` (reliability × boundary-distance factor) |
| `engine/decision.py` | Orchestrates extraction → heuristic → ML (if enabled/used) → rules → confidence → A/B arm; `rule_set_for_decision = RULE_SET_V2 if ML used else RULE_SET_V1` |
| `engine/explanation_builder.py` | Reads audit entries; builds summary, factors, timeline, SHAP contributions |
| `services/base.py` | Async HTTP; circuit breaker (CLOSED/OPEN/HALF_OPEN, 120s open timeout, probe lock); retry (3 attempts, exponential backoff, deterministic jitter); `classify_response` (408→TIMEOUT, ≥500→SERVICE_DOWN, 404→NO_RECORD, 400→FORMAT_ERROR); sync/async Redis dual support |
| `services/vault.py` | Optional Vault KV v2 read/write; `get_pii_encryption_key()`/`get_pan_salt()` with env fallback; `create_pii_service_from_vault_or_env()` — **available but not yet wired into API/worker** |
| `services/document_parser.py` | Deterministic parsing: bank statement (HDFC/ICICI/SBI keyword+regex), salary slip (regex earnings), GST filing (taxable/turnover/ITC); PDF via pdfplumber→PyPDF2, images via tesseract OCR; confidence scoring; returns `DocumentFeatures` (never raises on unparseable content) |
| `services/metrics.py` | Prometheus: application counts, task duration/failures, external latency, circuit state, decision confidence, drift alerts, A/B assignments, batch latency, prediction cache, MLflow runs |
| `ml/governance/drift_detector.py` | Two-sided KS test per numeric feature vs reference snapshot; `DriftDetectionReport.to_audit_payload()`; single-sample expansion ×32 |
| `ml/governance/ab_test.py` | `sha256(application_id) % 10000` bucket vs `ml_ratio` (default 0.10); outcome summaries with profit model |
| `ml/governance/model_registry.py` | File-backed JSON registry: `list_versions`, `get`, `latest`, `register_training_run`, `compare_versions`; deterministic (sorted keys); **registry file currently empty — XGB_V1 lives in `ml/models/manifest.yaml`** |
| `ml/causal/ab_framework.py` | ExperimentConfig/Result; deterministic arm assignment (sha256, 50/50, optional stratification by grade+purpose); Welch t-test; bootstrap CI (seeded); profit-based metrics |
| `ml/causal/champion_challenger.py` | Promotion/rollback verdict from ExperimentResult: significant + CI>0 → PROMOTE; significant + CI<0 → ROLLBACK; else CONTINUE; enforces min sample size on both arms |
| `ml/causal/psm.py` | Logistic-propensity (seeded) or heuristic propensity; greedy 1:1 nearest-neighbor matching within caliper; ATT + CI + balance (SMD/variance ratios) |
| `ml/causal/synthetic_control.py` | SLSQP (or grid) weight optimization on pre-treatment RMSE; synthetic counterfactual trajectory; causal effect = mean post-treatment gap |
| `ml/models/survival_coxph.py` | CoxPH: partial likelihood + Breslow ties, Newton–Raphson, hazard ratios, C-index, baseline hazard |
| `ml/models/survival_km.py` | Kaplan–Meier with Greenwood 95% bands; stratified curves; median survival |
| `ml/models/uplift_xgb.py` | Two-model uplift (treatment vs control XGB); uplift scores, segmentation, Qini coefficient |
| `ml/optimize/async_loader.py` | Background-thread model/SHAP loading; `wait_ready(timeout)`; explicit `_load_error` |
| `ml/optimize/onnx_export.py` | XGB→ONNX conversion + latency benchmark vs pickle |
| `ml/optimize/prediction_cache.py` | Thread-safe LRU+TTL; deterministic keys (`sha256(sorted features + model_version)`); `cached_explain` decorator |
| `ml/monitoring/drift_reporter.py` | Evidently `DataDriftTable` or KS fallback; HTML/JSON reports; Prometheus counters |
| `ml/portfolio/risk_aggregator.py` | Portfolio summary (buckets low≤25/medium≤50/high≤75/very_high>75), percentiles, stress tests, HHI concentration |
| `ml/portfolio/report_generator.py` | Markdown/HTML portfolio and stress-test reports |
| `ml/explain/llm_narrative.py` | PII-stripped narrative generation: remote LLM → local Ollama → deterministic template fallback; refusal detection; JSON schema; temperature 0.3 |
| `ml/explain/policy_rag.py` | Parses `docs/policy_corpus/*.md` into numbered snippets; TF-IDF or ChromaDB retrieval; `[source §section]` citations |
| `ml/experiment_tracking.py` | MLflow wrapper; no-op when `MLFLOW_ENABLED=false` (default) |
| `cli/portfolio.py` | `auditlend-portfolio summary|stress-test|html-report` |

### 5.3 How Modules Are Connected

- **API → Worker**: transactional outbox (application + outbox row committed together); outbox poller dispatches with deterministic task IDs (V1 noted a poller/direct-dispatch ambiguity — **in the current code the poller is the active path**, `worker/celery_app.py:45–50`).
- **API → Idempotency**: Redis fast path, Postgres durable fallback, SHA-256 payload hash, 409 on mismatch.
- **API → Encryption → Worker**: PII encrypted at rest (AES-256-GCM), decrypted by the worker (`_load_application_user_data`); key from env or (future) Vault.
- **Worker → External services**: `BaseExternalService` subclasses with circuit breaker + retry + typed `FailureType`; results cached in `external_data` and reused on redelivery.
- **Worker → Engine**: `compute_decision_from_env()` — the engine is pure; the worker supplies service results and user data.
- **Engine → ML**: `MLScorer` loads XGB_V1 from the manifest, calibrates, explains with SHAP, detects drift; low confidence → `fallback_used=True` → heuristic path (RULE_SET_V1) while still reporting `model_version: XGB_V1`.
- **Worker → Audit**: `write_audit_entry()` after every significant step; redaction (`_redact_user_data`, `audit_safe_features`) before snapshots.
- **Explanation → Audit**: the builder reads audit entries only.
- **Monitoring → Drift**: `POST /monitoring/drift` → `EvidentlyDriftReporter`/KS fallback → `drift_check` audit entry; worker-side drift alerts land in `DRIFT_DETECTED` audit entries and `drift_alerts_total` metrics.
- **Portfolio CLI ↔ engine/ML outputs**: `cli/portfolio.py` consumes decision JSON (the same shape produced by `/decision`) to generate summaries/stress tests/HTML.

---

## 6. Core Workflows

### 6.1 Happy Path (Unchanged Core, Updated Details)

#### 6.1.1 Client Submits Application

```bash
curl -X POST http://localhost:8000/api/v1/apply-loan \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-read-write" \
  -H "Idempotency-Key: v2-001" \
  -d '{
    "idempotency_key": "v2-001",
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
  }'
```

API steps (same as V1, plus auth/limits): composite auth → rate limit → validate → payload hash → idempotency check (Redis → Postgres) → encrypt PII → `pan_hash` → single-transaction insert (application + outbox + idempotency record) → cache response → HTTP 201.

#### 6.1.2 Worker Processes Application

1. **Dispatch**: outbox poller picks the PENDING row (`FOR UPDATE SKIP_LOCKED`) and calls `celery_app.send_task("worker.tasks.process_application.process_application", args=[application_id], task_id=f"{task_name}:{application_id}")`. The deterministic task ID makes duplicate polling safe.
2. **Claim** (`_claim_application`): atomic `UPDATE ... SET status='PROCESSING' WHERE id=:id AND (status='PENDING' OR (status='PROCESSING' AND updated_at < now() - 300s))`. Terminal → return stored result; locked-by-another → "being processed". Audit: `PROCESSING_STARTED`.
3. **Fetch** (`_fetch_external_data`): `asyncio.gather` over the three providers; each wrapped by `_fetch_or_reuse_external_data` (reuse `external_data` row if present; check-then-insert race guard). Each call runs the circuit breaker + retry + classification path in `services/base.py`.
4. **Decide** (`compute_decision_from_env`):
   - Heuristic score via `compute_risk_score()` (RULE_SET_V1 weights).
   - If `ML_ENABLED=true`: `MLScorer` loads XGB_V1, maps features, calibrates probability (isotonic), runs SHAP (top-8 contributions), checks drift against the reference snapshot. Low model confidence → `fallback_used=True` → heuristic score.
   - `rule_set_for_decision = RULE_SET_V2 if ml_result and ml_result.used else RULE_SET_V1`.
   - Rules evaluated in priority order; GST gate caps effective score at `approve_moderate_threshold - 1.0`.
   - `compute_data_reliability(failure_types, used_fallback_credit)` → penalties per `FailureType` (TIMEOUT 0.30, STALE_DATA 0.20, SERVICE_DOWN 0.30, PARTIAL_DATA 0.20, FORMAT_ERROR 0.30, PAN_MISMATCH 0.20, NO_RECORD 0.10; max penalty cap).
   - `compute_decision_confidence(risk_score, decision, data_reliability, failure_types)` = reliability × boundary-distance factor (APPROVE: ≥80→1.0, ≥70→0.9, ≥55→0.7, <55→0.6; DECLINE: ≤20→1.0, ≤34→0.85, >34→0.75; NEEDS_REVIEW 0.5).
   - Confidence < threshold (env `CONFIDENCE_THRESHOLD`, default 0.6) → override to `NEEDS_REVIEW`.
   - A/B arm via `assignment_from_env(application_id)` → `ab_test_arm` on output; metrics recorded.
5. **Store** (`_store_processing_results`): re-check terminal state under `FOR UPDATE`; update status/decision/confidence; persist `ExternalData` rows + `{SOURCE}_FETCH` audits; write `ML_SCORING`, `DRIFT_DETECTED` (when `alert_count > 0`), `DECISION_CALCULATION`, `MANUAL_REVIEW_OVERRIDE`/`MANUAL_REVIEW_ROUTING` entries; record A/B + drift metrics. All snapshots redacted.
6. **Return** dict: `application_id, status, decision, confidence, data_reliability, risk_score, rule_version, model_version, scoring_strategy, ab_test_arm`.

#### 6.1.3 Client Polls

```bash
curl -s -H "X-API-Key: dev-key-read-only" http://localhost:8000/api/v1/status/UUID
curl -s -H "X-API-Key: dev-key-read-only" http://localhost:8000/api/v1/decision/UUID
curl -s -H "X-API-Key: dev-key-read-only" http://localhost:8000/api/v1/explanation/UUID
# decision: {decision, confidence, data_reliability, risk_score, rule_version, scoring_strategy, model_version, ab_test_arm, ...}
# explanation: summary + timeline from audit entries + factor objects + SHAP contributions
```

### 6.2 Failure Scenarios (Unchanged Semantics)

- **Provider timeout** → retries exhausted → `FailureType.TIMEOUT` → fallback value + 0.30 reliability penalty → confidence drop → possibly `NEEDS_REVIEW`; audit records `CREDIT_BUREAU_FETCH` with `error_type=TIMEOUT`, `fallback_used=True`.
- **All providers down** → penalties stack (0.30+0.30+0.10 capped) → reliability ~0.30 → confidence < 0.6 → `MANUAL_REVIEW`.
- **GST PAN mismatch** → `gst_compliant=False` → GST gate caps effective score to 54.0 → `NEEDS_REVIEW`; explanation notes the data-quality issue.
- **Worker crash mid-processing** → claim includes stale-PROCESSING recovery (>300s) → external data reused from `external_data` → identical decision (idempotent terminal states).
- **Task timeout/system error** → `_mark_manual_review_after_system_error` sets MANUAL_REVIEW/NEEDS_REVIEW with `error_type=PIPELINE_TIMEOUT`/`SYSTEM_ERROR`; audit preserved; `task_failures` metric incremented.
- **ML forced low confidence** (`failure_flags.ml_model = "FORCE_LOW_CONFIDENCE"`) → heuristic fallback; output shows `scoring_strategy="heuristic"`, `model_version="XGB_V1"`.

### 6.3 New API Workflows (V2)

#### Document Parsing (`POST /api/v1/apply-loan/documents`)

- Auth + rate limited. Content-type whitelist (PDF/PNG/JPEG/plain text) → 415; >10MB → 413; empty → 400.
- `parse_document_bytes(content, filename, document_type)` (hint: `bank_statement | salary_slip | gst_filing`, otherwise auto-detected by keywords: GSTIN/GSTR, payslip, statement).
- Response: `{filename, document_type, features, confidence, warnings}` where `features` come from `extract_bank_statement_features()` (`income_stability_score`, `average_monthly_inflow/outflow`, `bounce_count`, `salary_regularity`, `emi_to_income_ratio`).
- **Important**: the document is not stored, no audit entry is written, and the endpoint does not feed the decision pipeline server-side. A client parses here, then supplies the features in `user_data.bank_statement` on a later apply call. Parsing is deterministic (regex/keyword; OCR for images); unreadable input returns `confidence=0.0` + warnings, never an exception.

#### Batch Prediction (`POST /api/v1/batch/predict`)

- Requires `xgboost` installed, else HTTP 424. `features: list[dict]` up to 1000 rows (truncated), processed in sub-batches of 50.
- **Scaffold caveat**: `_predict_batch` currently returns `{"prediction": 0.5, "model_version": ..., "features": ...}` for every row — it does not invoke the model. `GET /batch/status` returns `ml_available` and `PredictionCache` stats (hits/misses/size).
- The production-grade machinery it is meant to sit on top of already exists: `ml/optimize/prediction_cache.py` (deterministic keys bound to `model_version`), `async_loader.py`, `onnx_export.py`.

#### Monitoring (`/api/v1/monitoring/*`)

- `POST /monitoring/drift` — `{reference_data, candidate_data, drift_share_threshold=0.1}` → Evidently `DataDriftTable` (or KS fallback) → `{drift_share, drifted_features, feature_drift_scores, dataset_drift}` + structlog `drift_check`.
- `GET /monitoring/reports` — lists files under `monitoring/reports` (**no auth dependency** — a known gap, see §10).
- `POST /monitoring/reports/generate` — writes `drift_report.html` + `test_suite.html`; currently builds from empty DataFrames, so artifacts are stubs.

### 6.4 Governance Workflows (V2)

- **A/B experiment** (runtime): every decision gets a deterministic arm (`sha256(application_id) % 10000` vs `AB_TEST_ML_RATIO`); `ab_assignments_total` and `ab_decision_confidence` metrics; arm recorded in decision output and `DECISION_CALCULATION` audit payload.
- **Champion/challenger** (offline): `ExperimentFramework.analyze(records)` → Welch t-test + seeded bootstrap CI on simulated profit → `ChampionChallengerDecision` (PROMOTE/ROLLBACK/CONTINUE) enforcing `min_sample_size` on both arms.
- **Causal analysis** (offline): PSM for ATT with balance diagnostics; synthetic control for counterfactual trajectories; both deterministic.
- **Model registration**: `register_training_run(manifest_path, model_version, calibration_manifest_path)` upserts into `ml/governance/model_registry.json` (deterministic JSON, sorted keys, per-split metrics, calibration Brier/ECE, `data_hash`, feature counts). XGB_V1 is documented in `ml/models/manifest.yaml` but the registry file is currently uninitialized.
- **Survival/uplift** (offline): KM strata + CoxPH hazard ratios on portfolio data; two-model uplift with Qini.
- **Portfolio**: `auditlend-portfolio summary|stress-test|html-report` over decision JSON — buckets, percentiles, HHI, stress scenarios.

---

## 7. Data Layer & State Management

### 7.1 Database Schema (Unchanged Five Tables)

The schema documented in V1 is unchanged: `loan_applications`, `idempotency_records`, `outbox`, `external_data`, `audit_logs`. Key properties to retain:

- `loan_applications`: UUID id, `pan_hash` (salted SHA-256 — raw PAN never stored), `encrypted_user_data` BYTEA, `encryption_nonce`, status (PENDING/PROCESSING/COMPLETED/MANUAL_REVIEW), decision, confidence NUMERIC(3,2), failure_flags JSONB, indexes on status/idempotency_key/pan_hash.
- `idempotency_records`: key PK, application_id FK, response JSONB (includes `_request_hash`).
- `outbox`: BIGSERIAL id, task_name, task_args JSONB, status PENDING/PROCESSED/FAILED, `processed_at`, `error_message`; index (status, created_at).
- `external_data`: UNIQUE(application_id, source_type); request_params/response_data JSONB; failure_type; idempotency key per source (`f"{source_type.lower()}:{application_id}"`).
- `audit_logs`: append-only; step, input_snapshot/output_snapshot JSONB, error_type, fallback_used, fallback_reason, rule_version, actor, created_at; index (application_id, created_at); UPDATE/DELETE blocked by trigger (`20260429_0005_audit_protection.py`).

### 7.2 Storage and Caching Logic

- PII encrypted as one opaque blob; only `pan_hash` is queryable.
- Provider snapshots in `external_data` are reused for crash-recovery idempotency; audit snapshots are redacted (`_redact_user_data`, `audit_safe_features`).
- Redis: idempotency TTL 86400s; circuit breaker keys (`circuit:{service}:*`) with TTLs; Celery broker/backend.
- **New (V2) caching**: `PredictionCache` (LRU, default 1024 entries, TTL 3600s, thread-safe) with deterministic keys `sha256(sorted features + model_version)` — a replay can never cross model versions; `cached_explain` decorator stores `result.to_audit_payload()`.

### 7.3 Indexing, Optimization

Same indexes as V1; new optimization surfaces: ONNX export for batch inference, async background model loading, and the prediction cache. The ML artifacts (model pkl, calibrator pkl, features json, reference snapshot json, manifest) are versioned files under `ml/models/` and mounted into the worker container.

---

## 8. Key Design Patterns Used

### 8.1 Identify Patterns

| Pattern | Location | Why Used |
| --- | --- | --- |
| **Transactional Outbox** | `api/routes/applications.py` (application + outbox + idempotency in one transaction) | Exactly-once processing intent |
| **Idempotency (two-tier)** | Redis fast path + Postgres durable fallback + payload hash | Duplicate-free replay; 409 on same-key-different-payload |
| **Content-addressed task IDs** | `worker/outbox_poller.py` (`task_id=f"{task_name}:{application_id}"`) | Duplicate-safe dispatch even if polling races or redelivers |
| **Atomic Claiming** | `_claim_application` conditional UPDATE + stale-PROCESSING recovery (300s) | No duplicate processing across workers |
| **External Data Reuse** | `_fetch_or_reuse_external_data` | Crash-recovery idempotency; no redundant provider calls |
| **Circuit Breaker** | `services/base.py` (CLOSED/OPEN/HALF_OPEN, Redis-backed, 120s open timeout, probe lock) | Fail-fast; no cascading failures |
| **Retry + Exponential Backoff + Deterministic Jitter** | `_retry_delay` (`base·2^attempt + (attempt+1)·137%500/1000`) | Thundering herd prevention without breaking determinism |
| **Typed Failure Classification** | `FailureType` enum + `classify_response` | No unclassified external failures, ever |
| **Separation: risk score / data reliability / calibrated confidence** | `engine/scoring.py`, `engine/confidence.py` | Data quality must never be implicit |
| **Immutable Rule Sets** | `engine/rule_sets.py` frozen dataclasses; RULE_SET_V2 selected only when ML is used | No in-place rule mutation; version recorded in audit |
| **Append-Only Audit** | `models/audit_log.py` + DB trigger | Regulatory immutability |
| **Explanation from Audit Trail** | `engine/explanation_builder.py` | Describes what happened, not what would happen now |
| **Deterministic Mocks** | `mock_apis/*` | Reproducible integration tests |
| **Deterministic A/B Assignment** | `ml/governance/ab_test.py`, `ml/causal/ab_framework.py` (sha256 buckets) | Auditable experiments; no `random()` |
| **Seeded Statistical Methods** | bootstrap `Random(42)`, sklearn `random_state=42` | Reproducible CIs, PSM, uplift |
| **Fallback Chain** | service→fallback data, ML→heuristic, LLM→template, Evidently→KS, ChromaDB→TF-IDF | Graceful degradation, always flagged `fallback_used` and audited |
| **Prediction Cache bound to model version** | `ml/optimize/prediction_cache.py` | Cache replays can't cross model versions |
| **PII Boundary Stripping** | `ml/explain/llm_narrative.py` `_strip_pii` | Raw PII never leaves the system even for LLM calls |
| **Policy-Grounded Generation** | `ml/explain/policy_rag.py` `[source §section]` citations | LLM output traceable to the policy corpus |
| **Strangler-style optional deps** | capability flags (`VAULT_AVAILABLE`, `EVIDENTLY_AVAILABLE`, `_ML_AVAILABLE`, `MLFLOW_ENABLED`) | System works without any heavy dependency |

### 8.2 Where and Why They Are Used

Every pattern above maps to the project invariants (see `AGENTS.md`): determinism, idempotency, append-only audit, typed failures, no silent fallbacks, immutable rules, encrypted PII, explicit data reliability. The V2 additions extend the same invariants to the ML platform: seeded statistics keep experiments reproducible, the registry keeps model versions auditable, the prediction cache keys on model version, and the LLM narrative is the only sanctioned non-determinism — bounded by PII stripping, refusal detection, schema enforcement, and a deterministic fallback.

### 8.3 Benefits and Drawbacks in This Context

| Pattern | Benefit | Drawback |
| --- | --- | --- |
| Content-addressed task IDs | Duplicate-safe dispatch without Celery-level dedup | Task ID collides if the same app is re-enqueued intentionally — need outbox status discipline |
| Deterministic A/B assignment | Replayable, auditable arms | Bucketing on `application_id` alone is not stratified at runtime (runtime path uses `ab_test.py`; stratification lives in the offline framework) |
| Seeded statistics | Reproducible analysis | Results are only reproducible on the same library versions (XGBoost `hist` is deterministic within a build) |
| Prediction cache | Faster repeat predictions | TTL/staleness risk if features semantics change without a version bump |
| LLM narrative fallback chain | Always returns something | Fallback prose is generic; refusal detection is regex-based and can miss novel refusals |
| Optional deps everywhere | Runs anywhere | Capability flags multiply configuration surface; scaffold routes (batch) can mislead users about real capability |

---

## 9. Performance & Scalability Considerations

### 9.1 Bottlenecks in the Current System

| Bottleneck | Location | Impact | Evidence |
| --- | --- | --- | --- |
| Single PostgreSQL writer | all five tables | ~100 apps/s ceiling; write amplification per application | no replicas/pooling in compose |
| Provider latency worst case | `services/base.py` | 30s timeout × 3 retries per service, three services gathered | task timeout 60s → MANUAL_REVIEW |
| ML cold start | worker startup preload | slow container startup with XGBoost + calibrator + SHAP | `async_loader` exists but worker preloads at startup |
| Audit write fan-out | `_store_processing_results` | 5–10 synchronous inserts per application | no batching |
| Batch scaffold | `api/routes/batch.py` | constant 0.5 predictions; no real model path | `_predict_batch` stub |
| File-backed registry | `model_registry.py` | single-writer JSON; no locking | fine at this scale, not at multi-writer scale |

### 9.2 How It Scales (or Doesn't)

| Dimension | Current | At Scale |
| --- | --- | --- |
| Throughput | ~100 apps/s (single writer) | read replicas + PgBouncer + Kafka outbox |
| ML inference | in-process XGBoost/SHAP | ONNX + prediction cache + dedicated serving |
| Drift/monitoring | per-decision KS in worker + on-demand Evidently | streaming drift pipeline (e.g., Deequ/whylogs) |
| Experiments | deterministic buckets, offline analysis | feature-store backed online evaluation |
| Audit | unbounded growth | S3/object-store archival policy |

### 9.3 Suggestions for Improvement

1. Wire the batch endpoint to the real model path (`AsyncModelLoader.get_model` + `PredictionCache` + optional ONNX session) and return per-row SHAP/calibrated probabilities.
2. Batch audit writes (`executemany`).
3. Replace the file-backed registry with Postgres-backed registry rows (or adopt MLflow as the registry) so `model_version` lookups are transactional.
4. Add audit-log retention/archival policy (S3 after N days).
5. Runtime-stratified A/B assignment (grade+purpose hash like the offline framework) if arm imbalance is observed.
6. Per-decision-type confidence thresholds (approve vs decline) in RuleSet.
7. Authenticate `GET /monitoring/reports`.
8. Populate `UpliftResult.qini_coefficient` automatically in `fit`/`predict_uplift` (currently must be computed explicitly).
9. Replace the placeholder XGB_V1 reference snapshot (4 synthetic values/feature) with a real training distribution for meaningful KS drift.
10. Load the ML model lazily via `AsyncModelLoader` instead of eager startup preload to cut container startup time.

---

## 10. Weaknesses & Limitations

### 10.1 Design Flaws or Risks (Updated for V2)

| Issue | Severity | Description |
| --- | --- | --- |
| Batch predict is a scaffold | High | Returns constant 0.5; the endpoint implies ML capability that doesn't exist yet |
| `GET /api/v1/monitoring/reports` unauthenticated | High | Report files are listed without auth while every other route requires it |
| Vault is not wired | Medium | `create_pii_service_from_vault_or_env` is unused by API and worker; env vars remain the live path |
| Model registry empty | Medium | `model_registry.json` has no registered versions; XGB_V1 governance lives only in `manifest.yaml` |
| XGB_V1 reference snapshot is placeholder | Medium | 4 synthetic values per feature → KS drift alerts are advisory only |
| Single confidence threshold (0.6) | Medium | Not risk-tiered (unchanged from V1) |
| LLM narrative non-determinism | Medium | Intentional, but prose drift complicates regression testing; fallback template covers outages |
| No audit retention policy | High | Unbounded audit growth (unchanged from V1) |
| Duplicate PAN fraud gap | Medium | Same PAN hash with different identities is not flagged (unchanged from V1) |
| Static dev API keys | Medium | Rotation/scoping only partially addressed; OIDC mitigates for production clients |
| Runtime A/B not stratified | Low | Offline framework stratifies; runtime assignment is hash-only |
| Outbox poller single-process | Medium | One poller in compose; multiple pollers would need care (SKIP_LOCKED helps) |

### 10.2 Technical Debt Areas

| Area | Why It's Debt | Refactoring Suggestion |
| --- | --- | --- |
| `batch.py` stub | Predicts 0.5 regardless of features | Wire to `AsyncModelLoader`/`PredictionCache`/ONNX; add row-level audit of batch predictions |
| Empty-DataFrame monitoring report generation | Generates artifacts with no data | Accept actual reference/candidate data or remove the endpoint |
| Dead regexes in `document_parser.py` | `HDFC_TXN_RE`/`ICICI_TXN_RE`/`SBI_TXN_RE` compiled but unused | Remove or adopt |
| Vault factory unused | Code exists without integration | Wire `create_pii_service_from_vault_or_env` behind a `VAULT_URL` env gate |
| Hardcoded penalties/thresholds | `engine/confidence.py` values not configurable | Move into RuleSet-like config |
| Manifest is `.yaml` but JSON | `ml/models/manifest.yaml` contains JSON | Rename to `.json` or actually YAML it |
| `UpliftResult.qini_coefficient` never auto-populated | Silent None in result | Compute in `fit` |
| LLM refusal detection regexes | `_detect_refusal` pattern list may miss novel phrasings | Add structured refusal schema + retry logic |

### 10.3 What Would Break Under Scale or Edge Cases

- **Very long DTI (>1.0)** — clamped, declined by DTI rule (safe).
- **All providers down** — reliability ≈ 0.30, confidence < 0.6 → MANUAL_REVIEW (safe).
- **Redis unavailable** — circuit breaker degrades to always-CLOSED; idempotency falls back to Postgres (slower, safe).
- **Duplicate PAN with different identities** — not detected (fraud gap).
- **Batch > 1000 rows** — silently truncated, not rejected: callers may get incomplete results.
- **OCR garbage** — `confidence=0.0` + warnings; client must decide whether to proceed.
- **LLM endpoint down** — deterministic template fallback (safe, generic).
- **Multi-region / multiple pollers** — outbox + `SKIP_LOCKED` tolerate concurrent pollers, but the registry, rate limiter, and single Postgres do not span regions.

---

## 11. How to Improve This System

### 11.1 Concrete, Actionable Improvements

1. **Complete the batch path**: replace `_predict_batch` stub with `AsyncModelLoader` + `PredictionCache` + calibrated probabilities; add `latency_ms` and audit rows per batch; reject (not truncate) oversized inputs.
2. **Authenticate reports**: add `require_auth` to `GET /api/v1/monitoring/reports`.
3. **Wire Vault**: honor `VAULT_URL` in the API and worker when constructing `PIIService`; add `VAULT_URL`/`VAULT_TOKEN`/`VAULT_MOUNT_POINT` to `.env.example`.
4. **Register XGB_V1** in the model registry and refresh the reference snapshot from real training data.
5. **Per-decision-type confidence thresholds** in `RuleSet` (approve 0.6, decline 0.4, review 0.6).
6. **Audit archival**: after 90 days move to object storage; keep the trigger-protected table bounded.
7. **Stratified runtime A/B**: hash `application_id|grade|purpose` for runtime arms, matching the offline framework.
8. **Batch audit writes** in `_store_processing_results`.
9. **Fraud signal**: flag repeated `pan_hash` across applications in decision output and audits.
10. **Lazy ML loading**: use `AsyncModelLoader` for worker startup instead of eager preload.

### 11.2 Better Architectural Alternatives

1. **Kafka outbox** for throughput beyond ~100 apps/s, with the Postgres outbox retained for exactly-once intent.
2. **Postgres-backed registry** (or MLflow-as-registry) so model governance is transactional and multi-writer safe.
3. **Dedicated serving layer** (ONNX Runtime server) if batch workloads dominate; keep in-process cache for latency-critical paths.
4. **Streaming drift** (whylogs/Deequ) rather than per-decision KS + on-demand Evidently, for continuous monitoring at scale.
5. **External IDP enforcement** (OIDC-only) in production; dev API keys behind an env gate.

### 11.3 Refactoring Suggestions

| Current | Proposed | Why |
| --- | --- | --- |
| Batch stub → real model path | wire loader+cache+ONNX | deliver actual batch capability |
| Unauthenticated reports route | add auth dependency | close security asymmetry |
| Vault factory unused | wire behind env gate | real secret rotation path |
| Empty-registry JSON | seed with XGB_V1 run | model governance completeness |
| Placeholder snapshot | rebuild from training distribution | meaningful drift detection |
| Unused parser regexes | remove | dead code |
| `.yaml` JSON manifest | rename `.json` | honest file semantics |

---

## 12. Learning Notes (For a Developer)

### 12.1 Key Concepts to Study from This Project

1. **Determinism as a system property** — not just "no randomness": the jitter formula, A/B buckets, task IDs, cache keys, and seeds are all part of one reproducible whole. Trace `sha256` usage across `ab_test.py`, `prediction_cache.py`, and `outbox_poller.py`.
2. **Idempotency done right** — Redis fast path, Postgres durable fallback, payload hash, 409 semantics; plus crash-recovery via `external_data` reuse and stale-PROCESSING claims.
3. **Append-only audit end-to-end** — ORM model → `write_audit_entry()` → DB trigger; redaction before snapshots; explanation built only from these entries.
4. **Transactional outbox + content-addressed dispatch** — the outbox row carries the task name/args; the poller derives a deterministic Celery task ID. Compare with V1's ambiguity note: the poller is now the active path.
5. **Circuit breaker state machine** — CLOSED/OPEN/HALF_OPEN with TTLs, probe lock, 0-retry probes; and how it degrades when Redis is unavailable.
6. **Separation of risk score, reliability, confidence** — the invariants make degraded data quality explicit and measurable.
7. **Immutable rule sets + versioned scoring strategy** — RULE_SET_V2 is only used when ML is actually used; the rule version and model version ride along in every decision and audit row.
8. **ML governance without a heavy platform** — file-backed registry, deterministic A/B, KS drift, seeded stats; MLflow/Evidently are optional add-ons. Study how capability flags keep the system runnable with zero ML deps.
9. **Causal inference on decision data** — PSM (ATT, balance diagnostics), synthetic control (SLSQP counterfactual), bootstrap CIs with a fixed seed; champion/challenger promotion logic that refuses to declare winners on small samples.
10. **Survival & uplift** — CoxPH hazard ratios (Breslow ties, Newton–Raphson), KM with Greenwood bands, two-model uplift with Qini; all deterministic.
11. **Portfolio risk analytics** — risk buckets, percentiles, stress tests, HHI concentration; the `auditlend-portfolio` CLI is a clean example of a thin CLI over pure functions.
12. **Governed non-determinism** — the LLM narrative path: PII stripping boundary, refusal detection, schema-constrained output, deterministic fallback flagged `fallback_used`, policy RAG citations. This is the pattern to copy when any future component must be non-deterministic.
13. **Deterministic document parsing** — regex/keyword extraction with explicit confidence scores and warnings; graceful degradation (no exceptions on garbage input).
14. **Chaos/durability testing** — `tests/chaos/` covers circuit breaker, exactly-once, idempotency under load, retry exhaustion, worker crash recovery; `tests/integration/test_remediation_verification.py` verifies fixes stay fixed.

### 12.2 What Skills This Project Demonstrates

- Async Python (FastAPI + httpx + `asyncio.gather`/`asyncio.run` in Celery).
- SQLAlchemy 2.x async + Postgres (JSONB, UUID, triggers, `FOR UPDATE SKIP_LOCKED`, conditional UPDATE claims).
- Redis (idempotency cache, circuit breaker, Celery broker).
- Celery (task semantics, result backend, deterministic task IDs).
- Crypto (AES-256-GCM, salted SHA-256) and secrets handling (Vault pattern with env fallback).
- ML engineering (XGBoost, isotonic calibration, ECE, SHAP, KS drift, ONNX, model registry).
- Causal inference & statistics (PSM, synthetic control, bootstrap CI, t-tests, CoxPH, KM, Qini).
- API design & security (composite auth, OAuth2/OIDC, rate limiting, security headers, 2xx/4xx semantics, Problem Details).
- Observability (Prometheus metrics, structlog, health/drift dashboards).
- Testing at three levels (unit/integration/chaos) with determinism-focused assertions.

### 12.3 How to Replicate or Build Something Similar

1. **Start with the invariants** — write them down (deterministic, idempotent, auditable, typed failures, no silent fallbacks) before any code.
2. **Data model first**: application + idempotency + outbox + external_data + audit (trigger-protected).
3. **Idempotency + transactional outbox** next; content-addressed dispatch when the worker arrives.
4. **Pure decision engine** (`engine/`), then reliability/confidence separate from the risk score.
5. **Service clients** with retry/backoff/jitter + circuit breaker + `FailureType`.
6. **Audit every step**, then **explain from the audit trail** (never recompute).
7. **Add ML last**: trained model + calibrator + SHAP; version everything; make it optional with a deterministic fallback.
8. **Govern**: deterministic A/B, KS drift, registry; keep statistics seeded.
9. **Harden**: rate limiting, composite auth, security headers, size limits, secret management, monitoring endpoints.
10. **Extend analytics**: causal, survival, uplift, portfolio — as offline, deterministic analysis modules with CLI/report outputs, never in the decision path.

The critical principle remains: **every function in `engine/` (and every ML artifact, bucket, cache key, and task ID) must produce the same output for the same input** — or, if it cannot (LLM narratives), be explicitly governed and fall back deterministically.

---

## 13. Appendix: What Changed Between V1 and V2

| Area | V1 (STUDY_GUIDE.md) | V2 (current codebase) |
| --- | --- | --- |
| Dispatch | outbox poller vs direct `apply_async` ambiguity | Outbox poller is the active path; content-addressed task IDs |
| Auth | static `X-API-Key` only | CompositeAuth: API key + OAuth2/OIDC JWT; rate limiter; security headers; request size limits |
| API surface | apply/status/decision/explanation | + `/batch/predict`, `/batch/status`, `/apply-loan/documents`, `/monitoring/drift`, `/monitoring/reports[/generate]` |
| Secrets | env-var only | optional HashiCorp Vault (available; not yet wired) |
| Document intake | `user_data.bank_statement` client-supplied only | + `services/document_parser.py` (PDF/OCR/regex) with `/apply-loan/documents` endpoint (standalone, not stored) |
| ML runtime | ML scorer + SHAP + drift audit in worker | unchanged core + `rule_set_for_decision = RULE_SET_V2 if ML used` + A/B arm metrics + drift metrics |
| ML platform | train/evaluate/calibrate/explain/governance/benchmark | + causal (PSM, synthetic control, A/B framework, champion/challenger), survival (KM, CoxPH), uplift (UpliftXGB, Qini), optimize (ONNX, async loader, prediction cache), monitoring (Evidently), portfolio (aggregator, reports, CLI), MLflow tracking |
| Model state | XGB_V1 trained/calibrated | XGB_V1 in manifest (AUC-ROC 0.9757, ECE 0.0036, $68.3M simulated profit delta); file-backed registry exists but uninitialized |
| Governance docs | `docs/CALIBRATION.md` | + `docs/AB_TESTING.md`, `docs/SURVIVAL_ANALYSIS.md`, `docs/BUSINESS_IMPACT.md`, `docs/policy_corpus/{CREDIT_POLICY,RULE_GOVERNANCE}.md` |
| Explainability | audit-trail reconstruction + SHAP | + LLM narrative (governed non-determinism) + policy RAG citations |
| Testing | unit/integration/chaos | 437+ unit tests incl. causal, survival, uplift, portfolio, vault, document parsing, drift reporter, experiment tracking, batch API |
| Known gaps | no rate limiting, static keys, no archival, single threshold | rate limiting/OIDC added; new gaps: batch stub, unauthenticated reports route, vault unwired, empty registry, placeholder drift snapshot |

---

*Study Guide V2 was generated from a deep analysis of the AuditLend Intelligence Core codebase on 2026-08-03, as a follow-on to STUDY_GUIDE.md (V1, 2026-05-03). It reflects the codebase at that point in time, including the ML lifecycle platform and security hardening that post-date V1. The project is in active development; some implementation details may change over time.*
