from __future__ import annotations

import json
from pathlib import Path

from ml.governance.model_registry import ModelRegistry


def test_model_registry_registers_and_retrieves_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "model_registry.json")
    manifest_path = _write_manifest(tmp_path / "v1_manifest.json", run_id="20260503T100000Z-phase7a", auc_roc=0.81)
    calibration_path = _write_calibration_manifest(tmp_path / "v1_calibration.json", raw_ece=0.08, calibrated_ece=0.04)

    record = registry.register_training_run(
        manifest_path,
        model_version="XGB_V1",
        calibration_manifest_path=calibration_path,
    )

    loaded = registry.get("XGB_V1")

    assert record.model_version == "XGB_V1"
    assert loaded.selected_candidate == "lightgbm"
    assert loaded.calibration_metrics["test"]["calibrated_ece"] == 0.04
    assert loaded.to_audit_fields() == {
        "model_version": "XGB_V1",
        "selected_candidate": "lightgbm",
    }


def test_model_registry_compares_versions_by_split_metrics(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "model_registry.json")
    manifest_a = _write_manifest(tmp_path / "v1_manifest.json", run_id="20260503T100000Z-phase7a", auc_roc=0.81)
    manifest_b = _write_manifest(tmp_path / "v2_manifest.json", run_id="20260503T110000Z-phase7b", auc_roc=0.85)

    registry.register_training_run(manifest_a, model_version="XGB_V1")
    registry.register_training_run(manifest_b, model_version="XGB_V2")

    comparison = registry.compare_versions("XGB_V1", "XGB_V2")

    assert comparison.left_version == "XGB_V1"
    assert comparison.right_version == "XGB_V2"
    assert comparison.split_deltas["test"]["auc_roc"] == 0.04
    assert comparison.candidate_changed is False


def _write_manifest(path: Path, *, run_id: str, auc_roc: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": run_id,
                "selected_candidate": "lightgbm",
                "artifact_path": str(path.with_suffix(".pkl")),
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


def _write_calibration_manifest(path: Path, *, raw_ece: float, calibrated_ece: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "test": {
                        "raw_metrics": {"brier_score": 0.13},
                        "calibrated_metrics": {"brier_score": 0.11},
                        "raw_calibration": {"ece": raw_ece},
                        "calibrated_calibration": {"ece": calibrated_ece},
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
