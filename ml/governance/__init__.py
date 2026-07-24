"""Governance utilities such as drift detection and model registry."""

from ml.governance.ab_test import (
    ABTestReport,
    ArmSummary,
    DEFAULT_ML_TRAFFIC_RATIO,
    ExperimentAssignment,
    OutcomeRecord,
    assign_experiment_arm,
    assignment_from_env,
    summarize_outcomes,
)
from ml.governance.drift_detector import (
    DEFAULT_DRIFT_P_VALUE_THRESHOLD,
    DriftDetectionReport,
    FeatureDriftResult,
    build_reference_feature_snapshot,
    detect_feature_drift,
    detect_feature_drift_from_snapshot,
)
from ml.governance.model_registry import ModelRegistry, ModelRegistryRecord, ModelVersionComparison

try:
    from ml.experiment_tracking import ExperimentTracker, get_tracker, MLFLOW_AVAILABLE  # noqa: F811
except ImportError:
    pass

__all__ = [
    "ABTestReport",
    "ArmSummary",
    "DEFAULT_DRIFT_P_VALUE_THRESHOLD",
    "DEFAULT_ML_TRAFFIC_RATIO",
    "DriftDetectionReport",
    "ExperimentAssignment",
    "ExperimentTracker",
    "FeatureDriftResult",
    "MLFLOW_AVAILABLE",
    "ModelRegistry",
    "ModelRegistryRecord",
    "ModelVersionComparison",
    "OutcomeRecord",
    "assign_experiment_arm",
    "assignment_from_env",
    "build_reference_feature_snapshot",
    "detect_feature_drift",
    "detect_feature_drift_from_snapshot",
    "get_tracker",
    "summarize_outcomes",
]
