"""Tests for ONNX export utilities (ml/optimize/onnx_export.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.optimize.onnx_export import (
    ONNX_AVAILABLE,
    BenchmarkResult,
    InferenceBenchmark,
    _infer_feature_count,
    _infer_feature_names,
    _percentile,
    export_to_onnx,
    load_onnx_model,
    predict_with_onnx,
)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_p50_of_sorted_list(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == pytest.approx(3.0)

    def test_p0_returns_first(self):
        values = [10.0, 20.0, 30.0]
        assert _percentile(values, 0) == pytest.approx(10.0)

    def test_p100_returns_last(self):
        values = [10.0, 20.0, 30.0]
        assert _percentile(values, 100) == pytest.approx(30.0)

    def test_single_element(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)

    def test_unsorted_input_still_works(self):
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        assert _percentile(values, 0) == pytest.approx(1.0)
        assert _percentile(values, 100) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _infer_feature_count / _infer_feature_names
# ---------------------------------------------------------------------------


class TestInferFeatureHelpers:
    def test_infer_count_from_n_features_in_(self):
        model = MagicMock()
        model.n_features_in_ = 7
        assert _infer_feature_count(model) == 7

    def test_infer_count_from_feature_importances_(self):
        model = MagicMock(spec=[])
        model.feature_importances_ = [0.1, 0.2, 0.3]
        assert _infer_feature_count(model) == 3

    def test_infer_count_fallback_to_10(self):
        model = MagicMock(spec=[])
        assert _infer_feature_count(model) == 10

    def test_infer_names_from_feature_names_in_(self):
        model = MagicMock()
        model.feature_names_in_ = ["a", "b", "c"]
        assert _infer_feature_names(model) == ["a", "b", "c"]

    def test_infer_names_fallback_f_prefix(self):
        model = MagicMock(spec=[])
        model.feature_importances_ = [0.1, 0.2, 0.3]
        names = _infer_feature_names(model)
        assert names == ["f0", "f1", "f2"]

    def test_infer_names_fallback_count_10(self):
        model = MagicMock(spec=[])
        names = _infer_feature_names(model)
        assert len(names) == 10
        assert names[0] == "f0"


# ---------------------------------------------------------------------------
# export_to_onnx — ONNX not installed path
# ---------------------------------------------------------------------------


class TestExportToOnnxUnavailable:
    def test_returns_none_when_onnx_unavailable(self, tmp_path):
        with patch("ml.optimize.onnx_export.ONNX_AVAILABLE", False):
            result = export_to_onnx(MagicMock(), ["f1", "f2"], tmp_path / "model.onnx")
        assert result is None

    def test_no_file_written_when_onnx_unavailable(self, tmp_path):
        output = tmp_path / "model.onnx"
        with patch("ml.optimize.onnx_export.ONNX_AVAILABLE", False):
            export_to_onnx(MagicMock(), ["f1", "f2"], output)
        assert not output.exists()


# ---------------------------------------------------------------------------
# export_to_onnx — ONNX installed path (mocked)
# ---------------------------------------------------------------------------


class TestExportToOnnxAvailable:
    def test_calls_convert_and_saves(self, tmp_path):
        output = tmp_path / "subdir" / "model.onnx"
        mock_onnx_model = MagicMock()
        mock_onnx_module = MagicMock()
        mock_convert = MagicMock(return_value=mock_onnx_model)

        with (
            patch("ml.optimize.onnx_export.ONNX_AVAILABLE", True),
            patch.dict("ml.optimize.onnx_export.__dict__", {
                "convert_xgboost": mock_convert,
                "onnx": mock_onnx_module,
            }),
        ):
            import ml.optimize.onnx_export as mod
            # Temporarily inject the mocked symbols
            original_convert = getattr(mod, "convert_xgboost", None)
            original_onnx = getattr(mod, "onnx", None)
            mod.convert_xgboost = mock_convert
            mod.onnx = mock_onnx_module
            try:
                result = mod.export_to_onnx(MagicMock(), ["f1", "f2"], output)
            finally:
                if original_convert is not None:
                    mod.convert_xgboost = original_convert
                elif hasattr(mod, "convert_xgboost"):
                    del mod.convert_xgboost
                if original_onnx is not None:
                    mod.onnx = original_onnx
                elif hasattr(mod, "onnx"):
                    del mod.onnx

        mock_convert.assert_called_once()
        mock_onnx_module.checker.check_model.assert_called_once_with(mock_onnx_model)
        mock_onnx_module.save_model.assert_called_once_with(mock_onnx_model, str(output))
        assert result == str(output)

    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "deep" / "nested" / "model.onnx"
        mock_onnx_module = MagicMock()
        mock_convert = MagicMock(return_value=MagicMock())

        import ml.optimize.onnx_export as mod
        original_convert = getattr(mod, "convert_xgboost", None)
        original_onnx = getattr(mod, "onnx", None)
        mod.convert_xgboost = mock_convert
        mod.onnx = mock_onnx_module
        try:
            with patch("ml.optimize.onnx_export.ONNX_AVAILABLE", True):
                mod.export_to_onnx(MagicMock(), [], output)
        finally:
            if original_convert is not None:
                mod.convert_xgboost = original_convert
            elif hasattr(mod, "convert_xgboost"):
                del mod.convert_xgboost
            if original_onnx is not None:
                mod.onnx = original_onnx
            elif hasattr(mod, "onnx"):
                del mod.onnx

        assert output.parent.exists()


# ---------------------------------------------------------------------------
# load_onnx_model
# ---------------------------------------------------------------------------


class TestLoadOnnxModel:
    def test_returns_none_when_unavailable(self, tmp_path):
        with patch("ml.optimize.onnx_export.ONNX_AVAILABLE", False):
            result = load_onnx_model(tmp_path / "model.onnx")
        assert result is None

    def test_loads_session_when_available(self, tmp_path):
        fake_session = MagicMock()
        import ml.optimize.onnx_export as mod

        original_onnx = getattr(mod, "onnx", None)
        mod.onnx = MagicMock()  # satisfy ONNX_AVAILABLE guard

        import sys
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = fake_session
        sys.modules.setdefault("onnxruntime", mock_ort)

        try:
            with patch("ml.optimize.onnx_export.ONNX_AVAILABLE", True):
                result = load_onnx_model(tmp_path / "model.onnx")
        finally:
            if original_onnx is not None:
                mod.onnx = original_onnx
            elif hasattr(mod, "onnx"):
                del mod.onnx

        assert result is fake_session


# ---------------------------------------------------------------------------
# predict_with_onnx
# ---------------------------------------------------------------------------


class TestPredictWithOnnx:
    def test_calls_session_run(self):
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_output = MagicMock()
        mock_input.name = "input"
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.array([[0.1, 0.9]])]

        features = np.array([[1.0, 2.0, 3.0]])
        result = predict_with_onnx(mock_session, features)

        mock_session.run.assert_called_once()
        assert isinstance(result, np.ndarray)

    def test_output_is_ndarray(self):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="x")]
        mock_session.get_outputs.return_value = [MagicMock(name="y")]
        mock_session.run.return_value = [np.array([0.3, 0.7])]

        result = predict_with_onnx(mock_session, np.array([[1.0]]))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_inference_benchmark_defaults(self):
        b = InferenceBenchmark()
        assert b.mean == 0.0
        assert b.p50 == 0.0
        assert b.p95 == 0.0
        assert b.p99 == 0.0

    def test_benchmark_result_has_onnx_and_native(self):
        br = BenchmarkResult()
        assert isinstance(br.onnx, InferenceBenchmark)
        assert isinstance(br.native, InferenceBenchmark)
