"""Feature drift detection for AuditLend model governance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from scipy.stats import ks_2samp

from ml.models.train import MODEL_NUMERIC_FEATURES
from services.metrics import drift_alerts_total

DEFAULT_DRIFT_P_VALUE_THRESHOLD = 0.01


@dataclass(frozen=True)
class FeatureDriftResult:
    """Kolmogorov-Smirnov drift summary for one feature."""

    feature_name: str
    reference_count: int
    candidate_count: int
    ks_statistic: float
    p_value: float
    alert_triggered: bool
    reference_mean: float
    candidate_mean: float


@dataclass(frozen=True)
class DriftDetectionReport:
    """Serializable feature drift report for one scoring window."""

    model_version: str | None
    p_value_threshold: float
    total_features: int
    alert_count: int
    drifted_features: list[FeatureDriftResult]
    checked_features: list[FeatureDriftResult]

    def to_audit_payload(self) -> dict[str, Any]:
        """Return a JSON-safe warning payload for audit logging."""

        payload = asdict(self)
        payload["drifted_features"] = [asdict(row) for row in self.drifted_features]
        payload["checked_features"] = [asdict(row) for row in self.checked_features]
        return payload


def detect_feature_drift(
    reference_feature_rows: Sequence[dict[str, Any]],
    candidate_feature_rows: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str] | None = None,
    model_version: str | None = None,
    p_value_threshold: float = DEFAULT_DRIFT_P_VALUE_THRESHOLD,
    increment_metrics: bool = True,
) -> DriftDetectionReport:
    """Run KS drift checks across a numeric feature set."""

    selected_features = tuple(feature_names or MODEL_NUMERIC_FEATURES)
    checked_features: list[FeatureDriftResult] = []

    for feature_name in selected_features:
        reference_values = _feature_values(reference_feature_rows, feature_name)
        candidate_values = _feature_values(candidate_feature_rows, feature_name)
        if not reference_values or not candidate_values:
            continue

        statistic, p_value = ks_2samp(reference_values, candidate_values, alternative="two-sided", method="auto")
        result = FeatureDriftResult(
            feature_name=str(feature_name),
            reference_count=len(reference_values),
            candidate_count=len(candidate_values),
            ks_statistic=round(float(statistic), 6),
            p_value=round(float(p_value), 6),
            alert_triggered=float(p_value) < p_value_threshold,
            reference_mean=round(sum(reference_values) / len(reference_values), 6),
            candidate_mean=round(sum(candidate_values) / len(candidate_values), 6),
        )
        checked_features.append(result)

        if result.alert_triggered and increment_metrics:
            drift_alerts_total.labels(
                feature=result.feature_name,
                model_version=model_version or "unknown",
            ).inc()

    drifted_features = [row for row in checked_features if row.alert_triggered]
    return DriftDetectionReport(
        model_version=model_version,
        p_value_threshold=p_value_threshold,
        total_features=len(checked_features),
        alert_count=len(drifted_features),
        drifted_features=drifted_features,
        checked_features=checked_features,
    )


def build_reference_feature_snapshot(
    feature_rows: Iterable[dict[str, Any]],
    *,
    feature_names: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Extract reference feature distributions for later drift checks."""

    selected_features = tuple(feature_names or MODEL_NUMERIC_FEATURES)
    rows = list(feature_rows)
    return {
        str(feature_name): _feature_values(rows, feature_name)
        for feature_name in selected_features
    }


def detect_feature_drift_from_snapshot(
    reference_snapshot: dict[str, Sequence[float]],
    candidate_feature_rows: Sequence[dict[str, Any]],
    *,
    model_version: str | None = None,
    p_value_threshold: float = DEFAULT_DRIFT_P_VALUE_THRESHOLD,
    increment_metrics: bool = True,
) -> DriftDetectionReport:
    """Run drift checks using a precomputed reference distribution snapshot."""

    checked_features: list[FeatureDriftResult] = []
    for feature_name, reference_values in reference_snapshot.items():
        candidate_values = _feature_values(candidate_feature_rows, feature_name)
        cleaned_reference = [float(value) for value in reference_values]
        if not cleaned_reference or not candidate_values:
            continue

        statistic, p_value = ks_2samp(cleaned_reference, candidate_values, alternative="two-sided", method="auto")
        result = FeatureDriftResult(
            feature_name=str(feature_name),
            reference_count=len(cleaned_reference),
            candidate_count=len(candidate_values),
            ks_statistic=round(float(statistic), 6),
            p_value=round(float(p_value), 6),
            alert_triggered=float(p_value) < p_value_threshold,
            reference_mean=round(sum(cleaned_reference) / len(cleaned_reference), 6),
            candidate_mean=round(sum(candidate_values) / len(candidate_values), 6),
        )
        checked_features.append(result)

        if result.alert_triggered and increment_metrics:
            drift_alerts_total.labels(
                feature=result.feature_name,
                model_version=model_version or "unknown",
            ).inc()

    drifted_features = [row for row in checked_features if row.alert_triggered]
    return DriftDetectionReport(
        model_version=model_version,
        p_value_threshold=p_value_threshold,
        total_features=len(checked_features),
        alert_count=len(drifted_features),
        drifted_features=drifted_features,
        checked_features=checked_features,
    )


def _feature_values(feature_rows: Sequence[dict[str, Any]], feature_name: str) -> list[float]:
    values: list[float] = []
    for row in feature_rows:
        value = row.get(feature_name)
        if value is None:
            continue
        values.append(float(value))
    return values
