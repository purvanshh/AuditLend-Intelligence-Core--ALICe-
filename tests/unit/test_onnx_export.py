from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.optimize.onnx_export import (
    ONNX_AVAILABLE,
    benchmark_inference,
    export_to_onnx,
    load_onnx_model,
    predict_with_onnx,
)


class _FakeNativeModel:
    def __init__(self):
        self.n_features_in_ = 10
        self.feature_names_in_ = [f"f{i}" for i in range(10)]

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.3, 0.7]])


def test_export_to_onnx_graceful_degradation(tmp_path: Path) -> None:
    if ONNX_AVAILABLE:
        return
    model = _FakeNativeModel()
    result = export_to_onnx(model, ["f0", "f1"], tmp_path / "model.onnx")
    assert result is None


def test_load_onnx_graceful_degradation(tmp_path: Path) -> None:
    if ONNX_AVAILABLE:
        return
    result = load_onnx_model(tmp_path / "nonexistent.onnx")
    assert result is None


def test_predict_with_onnx_requires_session() -> None:
    if ONNX_AVAILABLE:
        return
    features = np.zeros((1, 10), dtype=np.float32)
    try:
        predict_with_onnx(None, features)
        assert False, "should have raised"
    except (AttributeError, TypeError):
        pass


def test_benchmark_inference_graceful_degradation(tmp_path: Path) -> None:
    if ONNX_AVAILABLE:
        return
    model_path = tmp_path / "model.pkl"
    result = benchmark_inference(model_path, num_runs=5)
    assert result is None


def test_benchmark_inference_returns_stats_dict(tmp_path: Path) -> None:
    import pickle

    model = _FakeNativeModel()
    model_path = tmp_path / "test_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    if ONNX_AVAILABLE:
        result = benchmark_inference(model_path, num_runs=5)
        assert result is not None
        assert "onnx" in result
        assert "native" in result
        for backend in ("onnx", "native"):
            for key in ("mean", "p50", "p95", "p99"):
                assert key in result[backend]
                assert isinstance(result[backend][key], float)
    else:
        result = benchmark_inference(model_path, num_runs=5)
        assert result is None
