from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

try:
    import onnx
    import onnxmltools
    from onnxmltools.convert import convert_xgboost

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


logger = structlog.get_logger()


def export_to_onnx(
    model: Any,
    feature_names: list[str],
    output_path: str | Path,
) -> str | None:
    if not ONNX_AVAILABLE:
        logger.warning("onnx_export_unavailable", message="ONNX deps not installed, skipping export")
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    onnx_model = convert_xgboost(model, feature_names=feature_names)
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, str(output))
    logger.info("onnx_export_complete", path=str(output))
    return str(output)


def load_onnx_model(model_path: str | Path) -> Any | None:
    if not ONNX_AVAILABLE:
        logger.warning("onnx_load_unavailable", message="ONNX deps not installed, falling back to native")
        return None

    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session


def predict_with_onnx(onnx_session: Any, features: np.ndarray) -> np.ndarray:
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: features.astype(np.float32)})
    return np.asarray(result[0])


@dataclass
class InferenceBenchmark:
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


@dataclass
class BenchmarkResult:
    onnx: InferenceBenchmark = field(default_factory=InferenceBenchmark)
    native: InferenceBenchmark = field(default_factory=InferenceBenchmark)


def _percentile(values: list[float], p: float) -> float:
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * p / 100)))
    return sorted_values[index]


def benchmark_inference(
    model_path: str | Path,
    num_runs: int = 100,
) -> dict[str, dict[str, float]] | None:
    import pickle

    if not ONNX_AVAILABLE:
        logger.warning("onnx_benchmark_unavailable", message="ONNX deps not installed, skipping benchmark")
        return None

    with Path(model_path).open("rb") as handle:
        native_model = pickle.load(handle)

    sample_features = np.zeros((1, _infer_feature_count(native_model)), dtype=np.float32)

    native_latencies: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        native_model.predict_proba(sample_features)
        native_latencies.append((time.perf_counter() - start) * 1000)

    onnx_path = export_to_onnx(native_model, _infer_feature_names(native_model), str(Path(model_path).with_suffix(".onnx")))
    if onnx_path is None:
        return None

    onnx_session = load_onnx_model(onnx_path)
    onnx_latencies: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predict_with_onnx(onnx_session, sample_features)
        onnx_latencies.append((time.perf_counter() - start) * 1000)

    def _stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": round(float(np.mean(values)), 4),
            "p50": round(_percentile(values, 50), 4),
            "p95": round(_percentile(values, 95), 4),
            "p99": round(_percentile(values, 99), 4),
        }

    return {
        "onnx": _stats(onnx_latencies),
        "native": _stats(native_latencies),
    }


def _infer_feature_count(model: Any) -> int:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "feature_importances_"):
        return len(model.feature_importances_)
    return 10


def _infer_feature_names(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [f"f{i}" for i in range(_infer_feature_count(model))]
