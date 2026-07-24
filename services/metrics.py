from prometheus_client import Counter, Gauge, Histogram


loan_applications_total = Counter(
    "auditlend_applications_total",
    "Total loan applications",
    ["status"],
)

external_api_requests = Counter(
    "auditlend_external_api_requests_total",
    "External API calls",
    ["service", "status"],
)

external_api_latency = Histogram(
    "auditlend_external_api_latency_seconds",
    "External API call latency",
    ["service"],
)

circuit_breaker_state = Gauge(
    "auditlend_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"],
)

decision_confidence = Histogram(
    "auditlend_decision_confidence",
    "Decision confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

task_duration = Histogram(
    "auditlend_task_duration_seconds",
    "Celery task duration",
    ["task_name"],
)

task_failures = Counter(
    "auditlend_task_failures_total",
    "Celery task failures",
    ["task_name", "error_type"],
)

mlflow_runs_total = Counter(
    "auditlend_mlflow_runs_total",
    "Total MLflow experiment runs",
    ["status"],
)

mlflow_runs_failed_total = Counter(
    "auditlend_mlflow_runs_failed_total",
    "Failed MLflow experiment runs",
    ["error_type"],
)

drift_alerts_total = Counter(
    "auditlend_drift_alerts_total",
    "Feature drift alerts raised by the ML governance layer",
    ["feature", "model_version"],
)

ab_assignments_total = Counter(
    "auditlend_ab_assignments_total",
    "A/B experiment assignments by arm",
    ["arm"],
)

ab_decisions_total = Counter(
    "auditlend_ab_decisions_total",
    "A/B experiment decision outcomes by arm and scoring strategy",
    ["arm", "decision", "scoring_strategy"],
)

rate_limit_exceeded_total = Counter(
    "auditlend_rate_limit_exceeded_total",
    "Rate limit exceeded",
    ["client_key"],
)

auth_attempts_total = Counter(
    "auditlend_auth_attempts_total",
    "Auth attempts",
    ["method", "result"],
)

ab_decision_confidence = Histogram(
    "auditlend_ab_decision_confidence",
    "Decision confidence scores grouped by A/B arm",
    ["arm"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

drift_reports_generated_total = Counter(
    "auditlend_drift_reports_total",
    "Drift reports generated",
    ["format"],
)

drift_alerts_evidently_total = Counter(
    "auditlend_drift_alerts_evidently_total",
    "Evidently drift alerts",
    ["feature"],
)


prediction_cache_hits_total = Counter(
    "auditlend_prediction_cache_hits_total",
    "Prediction cache hits",
    [],
)

prediction_cache_misses_total = Counter(
    "auditlend_prediction_cache_misses_total",
    "Prediction cache misses",
    [],
)

prediction_cache_size = Gauge(
    "auditlend_prediction_cache_size",
    "Prediction cache entries",
    [],
)

batch_prediction_duration_seconds = Histogram(
    "auditlend_batch_prediction_duration_seconds",
    "Batch prediction duration",
    ["batch_size"],
)


def circuit_state_value(state: str) -> int:
    return {
        "CLOSED": 0,
        "OPEN": 1,
        "HALF_OPEN": 2,
    }.get(state, 0)
