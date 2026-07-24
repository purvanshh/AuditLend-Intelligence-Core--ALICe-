from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.experiment_tracking import ExperimentTracker, MLFLOW_AVAILABLE, get_tracker


def test_mflow_available_flag() -> None:
    from ml.experiment_tracking import MLFLOW_AVAILABLE as flag
    assert flag is False


def test_experiment_tracker_init_without_mlflow(tmp_path: Path) -> None:
    tracker = ExperimentTracker(
        tracking_uri="http://localhost:5000",
        experiment_name="test_experiment",
        registry_path=str(tmp_path / "registry.json"),
    )
    assert tracker.experiment_name == "test_experiment"
    assert tracker.registry.registry_path == tmp_path / "registry.json"
    assert tracker._enabled is False


def test_experiment_tracker_init_with_mlflow_enabled(tmp_path: Path) -> None:
    with patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True), patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow:
        tracker = ExperimentTracker(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        assert tracker.experiment_name == "test_experiment"
        assert tracker._enabled is True
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")


def test_start_run_noop_when_mlflow_unavailable(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    with tracker.start_run(run_name="test_run") as run:
        assert run is None


def test_start_run_with_mlflow(tmp_path: Path) -> None:
    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "test-run-id"

    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_active_run
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        with tracker.start_run(run_name="test_run", tags={"env": "test"}) as run:
            assert run is mock_active_run
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run", tags={"env": "test"}, nested=False)


def test_log_params_noop_when_mlflow_unavailable(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    tracker.log_params_from_dict({"learning_rate": 0.1, "max_depth": 6})
    tracker.log_metrics_from_dict({"auc_roc": 0.85})


def test_log_params_with_mlflow(tmp_path: Path) -> None:
    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        tracker.log_params_from_dict({"learning_rate": 0.1, "max_depth": 6})
        mock_mlflow.log_params.assert_called_once_with({"learning_rate": 0.1, "max_depth": 6})

        tracker.log_metrics_from_dict({"auc_roc": 0.85, "auc_pr": 0.82}, step=1)
        mock_mlflow.log_metrics.assert_called_once_with({"auc_roc": 0.85, "auc_pr": 0.82}, step=1)


def test_log_artifact_noop_when_mlflow_unavailable(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    tracker.log_artifact("/tmp/nonexistent.txt")


def test_log_artifact_with_mlflow(tmp_path: Path) -> None:
    artifact_file = tmp_path / "test_artifact.txt"
    artifact_file.write_text("test")
    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        tracker.log_artifact(str(artifact_file))
        mock_mlflow.log_artifact.assert_called_once_with(str(artifact_file))


def test_log_model_noop_when_mlflow_unavailable(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    tracker.log_model(None, "model")


def test_log_model_with_mlflow(tmp_path: Path) -> None:
    mock_model = MagicMock()
    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        tracker.log_model(mock_model, "xgboost_model", signature="sig", input_example=[1.0])
        mock_mlflow.xgboost.log_model.assert_called_once_with(
            mock_model, "xgboost_model", signature="sig", input_example=[1.0]
        )


def test_register_version_creates_file_registry_record(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    manifest_path = _write_manifest(tmp_path / "v1_manifest.json")

    record = tracker.register_version("XGB_V1", manifest_path)

    assert record.model_version == "XGB_V1"
    assert record.selected_candidate == "lightgbm"

    versions = tracker.list_versions()
    assert len(versions) == 1
    assert versions[0].model_version == "XGB_V1"


def test_register_version_with_mlflow_tags_run(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "v1_manifest.json")
    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        record = tracker.register_version("XGB_V1", manifest_path)
        assert record.model_version == "XGB_V1"
        mock_mlflow.set_tag.assert_any_call("model_version", "XGB_V1")
        mock_mlflow.set_tag.assert_any_call("selected_candidate", "lightgbm")


def test_load_model_falls_back_to_file_when_mlflow_unavailable(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    manifest_path = _write_manifest(
        tmp_path / "v1_manifest.json",
        artifact_path=str(tmp_path / "nonexistent.pkl"),
    )
    tracker.register_version("XGB_V1", manifest_path)
    with pytest.raises(FileNotFoundError):
        tracker.load_model("XGB_V1")


def test_load_model_with_mlflow_uses_model_registry(tmp_path: Path) -> None:
    mock_model = MagicMock()
    manifest_path = _write_manifest(
        tmp_path / "v1_manifest.json",
        artifact_path=str(tmp_path / "model.pkl"),
    )
    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        mock_mlflow.xgboost.load_model.return_value = mock_model
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        tracker.register_version("XGB_V1", manifest_path)
        model = tracker.load_model("XGB_V1", model_uri="models:/XGB_V1/latest")
        assert model is mock_model
        mock_mlflow.xgboost.load_model.assert_called_once_with("models:/XGB_V1/latest")


def test_search_runs_empty_when_mlflow_unavailable(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    runs = tracker.search_runs(max_results=5)
    assert runs == []


def test_search_runs_with_mlflow(tmp_path: Path) -> None:
    with (
        patch("ml.experiment_tracking.MLFLOW_AVAILABLE", True),
        patch("ml.experiment_tracking.mlflow", create=True) as mock_mlflow,
    ):
        mock_mlflow.search_runs.return_value = ["run1", "run2"]
        tracker = ExperimentTracker(
            registry_path=str(tmp_path / "registry.json"),
            enabled=True,
        )
        runs = tracker.search_runs(max_results=5)
        assert runs == ["run1", "run2"]
        mock_mlflow.search_runs.assert_called_once_with(max_results=5)


def test_get_tracker_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.experiment_tracking._tracker", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom_experiment")
    monkeypatch.setenv("MLFLOW_ENABLED", "false")

    try:
        tracker_a = get_tracker()
        tracker_b = get_tracker()
        assert tracker_a is tracker_b
        assert tracker_a.tracking_uri == "http://custom:5000"
        assert tracker_a.experiment_name == "custom_experiment"
    finally:
        monkeypatch.undo()


def test_get_tracker_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.experiment_tracking._tracker", None)
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_ENABLED"):
        monkeypatch.delenv(key, raising=False)

    try:
        tracker = get_tracker()
        assert tracker.tracking_uri == "http://localhost:5000"
        assert tracker.experiment_name == "auditlend"
    finally:
        monkeypatch.undo()


def test_list_versions_empty(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    assert tracker.list_versions() == []


def test_list_versions_after_registration(tmp_path: Path) -> None:
    tracker = ExperimentTracker(registry_path=str(tmp_path / "registry.json"))
    v1 = _write_manifest(tmp_path / "v1.json", run_id="v1", auc_roc=0.81)
    v2 = _write_manifest(tmp_path / "v2.json", run_id="v2", auc_roc=0.85)

    tracker.register_version("V1", v1)
    tracker.register_version("V2", v2)

    versions = tracker.list_versions()
    assert len(versions) == 2
    assert versions[0].model_version == "V1"
    assert versions[1].model_version == "V2"


def _write_manifest(
    path: Path,
    *,
    run_id: str = "20260503T100000Z-phase7a",
    auc_roc: float = 0.81,
    artifact_path: str | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": run_id,
                "selected_candidate": "lightgbm",
                "artifact_path": artifact_path or str(path.with_suffix(".pkl")),
                "data_hash": "abc123",
                "feature_count": 42,
                "split_counts": {"train": 100, "validation": 50, "test": 25},
                "metrics": {
                    "train": {
                        "auc_roc": 0.9,
                        "auc_pr": 0.85,
                        "brier_score": 0.1,
                        "positive_rate": 0.2,
                        "row_count": 100,
                    },
                    "validation": {
                        "auc_roc": auc_roc - 0.01,
                        "auc_pr": 0.8,
                        "brier_score": 0.12,
                        "positive_rate": 0.2,
                        "row_count": 50,
                    },
                    "test": {
                        "auc_roc": auc_roc,
                        "auc_pr": 0.79,
                        "brier_score": 0.13,
                        "positive_rate": 0.2,
                        "row_count": 25,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
