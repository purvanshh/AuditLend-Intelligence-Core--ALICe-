"""File-backed registry for versioned AuditLend ML model artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.models.evaluate import load_manifest


@dataclass(frozen=True)
class ModelRegistryRecord:
    """Serializable metadata for one registered model version."""

    model_version: str
    created_at: str
    selected_candidate: str
    manifest_path: str
    artifact_path: str
    calibration_manifest_path: str | None
    calibrator_artifact_path: str | None
    data_hash: str
    feature_count: int
    split_counts: dict[str, int]
    metrics: dict[str, dict[str, float | int]]
    calibration_metrics: dict[str, dict[str, Any]]

    def to_audit_fields(self) -> dict[str, Any]:
        """Return the minimal audit payload fields needed during ML scoring."""

        return {
            "model_version": self.model_version,
            "selected_candidate": self.selected_candidate,
        }


@dataclass(frozen=True)
class ModelVersionComparison:
    """Side-by-side metric comparison for two registered model versions."""

    left_version: str
    right_version: str
    split_deltas: dict[str, dict[str, float]]
    candidate_changed: bool


class ModelRegistry:
    """Persist and query versioned model metadata in a JSON registry file."""

    def __init__(self, registry_path: str | Path = "ml/governance/model_registry.json") -> None:
        self.registry_path = Path(registry_path)

    def list_versions(self) -> list[ModelRegistryRecord]:
        """Return all registered versions in deterministic order."""

        payload = self._load_registry_payload()
        records = [self._record_from_dict(row) for row in payload.get("models", [])]
        return sorted(records, key=lambda row: (row.created_at, row.model_version))

    def get(self, model_version: str) -> ModelRegistryRecord:
        """Look up a registered model version."""

        for record in self.list_versions():
            if record.model_version == model_version:
                return record
        raise KeyError(f"Model version '{model_version}' is not registered.")

    def latest(self) -> ModelRegistryRecord | None:
        """Return the newest registered model version, if any."""

        records = self.list_versions()
        return records[-1] if records else None

    def register_training_run(
        self,
        manifest_path: str | Path,
        *,
        model_version: str | None = None,
        calibration_manifest_path: str | Path | None = None,
    ) -> ModelRegistryRecord:
        """Upsert one Phase 3/5 training run into the registry."""

        manifest = load_manifest(manifest_path)
        resolved_model_version = model_version or str(manifest["run_id"])
        calibration_manifest = self._load_optional_calibration_manifest(
            calibration_manifest_path or self._default_calibration_manifest_path(manifest)
        )

        record = ModelRegistryRecord(
            model_version=resolved_model_version,
            created_at=str(manifest.get("created_at") or manifest["run_id"]),
            selected_candidate=str(manifest["selected_candidate"]),
            manifest_path=str(Path(manifest_path)),
            artifact_path=str(manifest["artifact_path"]),
            calibration_manifest_path=(
                str(calibration_manifest_path or self._default_calibration_manifest_path(manifest))
                if calibration_manifest
                else None
            ),
            calibrator_artifact_path=(
                str(calibration_manifest.get("calibrator_artifact_path"))
                if calibration_manifest
                else None
            ),
            data_hash=str(manifest["data_hash"]),
            feature_count=int(manifest["feature_count"]),
            split_counts={key: int(value) for key, value in manifest["split_counts"].items()},
            metrics={
                split_name: {
                    key: value
                    for key, value in split_metrics.items()
                }
                for split_name, split_metrics in manifest["metrics"].items()
            },
            calibration_metrics=self._extract_calibration_metrics(calibration_manifest),
        )
        self._upsert_record(record)
        return record

    def compare_versions(self, left_version: str, right_version: str) -> ModelVersionComparison:
        """Compare two registered versions split by split."""

        left = self.get(left_version)
        right = self.get(right_version)
        split_names = sorted(set(left.metrics) | set(right.metrics))

        split_deltas: dict[str, dict[str, float]] = {}
        for split_name in split_names:
            left_metrics = left.metrics.get(split_name, {})
            right_metrics = right.metrics.get(split_name, {})
            metric_names = sorted(set(left_metrics) | set(right_metrics))
            split_deltas[split_name] = {
                metric_name: round(
                    float(right_metrics.get(metric_name, 0.0)) - float(left_metrics.get(metric_name, 0.0)),
                    6,
                )
                for metric_name in metric_names
            }

        return ModelVersionComparison(
            left_version=left_version,
            right_version=right_version,
            split_deltas=split_deltas,
            candidate_changed=left.selected_candidate != right.selected_candidate,
        )

    def _upsert_record(self, record: ModelRegistryRecord) -> None:
        payload = self._load_registry_payload()
        rows = [row for row in payload.get("models", []) if row.get("model_version") != record.model_version]
        rows.append(asdict(record))
        rows.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("model_version", ""))))
        payload["models"] = rows
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _load_registry_payload(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"models": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _record_from_dict(self, payload: dict[str, Any]) -> ModelRegistryRecord:
        return ModelRegistryRecord(
            model_version=str(payload["model_version"]),
            created_at=str(payload["created_at"]),
            selected_candidate=str(payload["selected_candidate"]),
            manifest_path=str(payload["manifest_path"]),
            artifact_path=str(payload["artifact_path"]),
            calibration_manifest_path=payload.get("calibration_manifest_path"),
            calibrator_artifact_path=payload.get("calibrator_artifact_path"),
            data_hash=str(payload["data_hash"]),
            feature_count=int(payload["feature_count"]),
            split_counts={key: int(value) for key, value in payload.get("split_counts", {}).items()},
            metrics={
                str(split_name): {str(key): value for key, value in split_metrics.items()}
                for split_name, split_metrics in payload.get("metrics", {}).items()
            },
            calibration_metrics={
                str(split_name): {str(key): value for key, value in split_metrics.items()}
                for split_name, split_metrics in payload.get("calibration_metrics", {}).items()
            },
        )

    def _default_calibration_manifest_path(self, manifest: dict[str, Any]) -> Path:
        return Path(manifest["artifact_path"]).parent / "isotonic_calibration_manifest.json"

    def _load_optional_calibration_manifest(self, calibration_manifest_path: str | Path | None) -> dict[str, Any] | None:
        if calibration_manifest_path is None:
            return None
        path = Path(calibration_manifest_path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _extract_calibration_metrics(self, calibration_manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if calibration_manifest is None:
            return {}

        extracted: dict[str, dict[str, Any]] = {}
        for split_name, split_payload in calibration_manifest.get("splits", {}).items():
            extracted[str(split_name)] = {
                "raw_brier_score": split_payload.get("raw_metrics", {}).get("brier_score"),
                "raw_ece": split_payload.get("raw_calibration", {}).get("ece"),
                "calibrated_brier_score": split_payload.get("calibrated_metrics", {}).get("brier_score"),
                "calibrated_ece": split_payload.get("calibrated_calibration", {}).get("ece"),
            }
        return extracted
