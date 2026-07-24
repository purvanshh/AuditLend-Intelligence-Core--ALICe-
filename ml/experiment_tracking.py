"""Optional MLflow experiment tracking wrapper around the ModelRegistry."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import structlog

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ml.governance.model_registry import ModelRegistry, ModelRegistryRecord
from services.metrics import mlflow_runs_failed_total, mlflow_runs_total


logger = structlog.get_logger(__name__)


class ExperimentTracker:
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        registry_path: str | Path = "ml/governance/model_registry.json",
        enabled: bool | None = None,
    ) -> None:
        self.registry = ModelRegistry(registry_path)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or "auditlend"
        self._enabled = MLFLOW_AVAILABLE if enabled is None else (enabled and MLFLOW_AVAILABLE)
        if self._enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Iterator[Any]:
        if self._enabled:
            try:
                with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
                    mlflow_runs_total.labels(status="started").inc()
                    yield run
            except Exception:
                mlflow_runs_failed_total.labels(error_type="start_run").inc()
                raise
        else:
            yield None

    def log_params_from_dict(self, params: dict[str, Any]) -> None:
        if self._enabled:
            mlflow.log_params(params)

    def log_metrics_from_dict(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        if self._enabled:
            mlflow.log_artifact(local_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Any = None,
        input_example: Any = None,
    ) -> None:
        if self._enabled:
            mlflow.xgboost.log_model(model, artifact_path, signature=signature, input_example=input_example)

    def register_version(
        self,
        model_version: str,
        manifest_path: str | Path,
        calibration_manifest_path: str | Path | None = None,
    ) -> ModelRegistryRecord:
        record = self.registry.register_training_run(
            manifest_path,
            model_version=model_version,
            calibration_manifest_path=calibration_manifest_path,
        )
        if self._enabled:
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("selected_candidate", record.selected_candidate)
            mlflow_runs_total.labels(status="registered").inc()
        return record

    def list_versions(self) -> list[ModelRegistryRecord]:
        return self.registry.list_versions()

    def load_model(self, model_version: str, model_uri: str | None = None) -> Any:
        if self._enabled and model_uri:
            return mlflow.xgboost.load_model(model_uri)
        record = self.registry.get(model_version)
        from ml.models.evaluate import load_model_artifact
        return load_model_artifact(record.artifact_path)

    def search_runs(self, max_results: int = 10) -> list[Any]:
        if not self._enabled:
            return []
        return mlflow.search_runs(max_results=max_results)


_tracker: ExperimentTracker | None = None


def get_tracker() -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "auditlend")
        registry_path = os.getenv("MLFLOW_REGISTRY_PATH", "ml/governance/model_registry.json")
        enabled = os.getenv("MLFLOW_ENABLED", "false").lower() in ("true", "1", "yes")
        _tracker = ExperimentTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            registry_path=registry_path,
            enabled=enabled,
        )
    return _tracker


__all__ = ["ExperimentTracker", "get_tracker", "MLFLOW_AVAILABLE"]
