from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pytest

from ml.optimize.async_loader import AsyncModelLoader


class _FakeModel:
    def predict_proba(self, X):
        return [[0.2, 0.8]]


def _write_fake_manifest(tmp_path: Path, model_path: str) -> Path:
    manifest = {
        "model_version": "XGB_V1",
        "model_artifact_path": model_path,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return manifest_path


def test_loader_init_and_is_ready(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(_FakeModel(), handle)
    manifest_path = _write_fake_manifest(tmp_path, str(model_path))

    loader = AsyncModelLoader(manifest_path, preload=True)
    ready = loader.wait_ready(timeout=10.0)
    assert ready
    assert loader.is_ready()


def test_loader_is_not_ready_before_load(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(_FakeModel(), handle)
    manifest_path = _write_fake_manifest(tmp_path, str(model_path))

    loader = AsyncModelLoader(manifest_path, preload=False)
    assert not loader.is_ready()

    loader._load_all()
    assert loader.is_ready()


def test_wait_ready_timeout(tmp_path: Path) -> None:
    loader = AsyncModelLoader(tmp_path / "nonexistent.json", preload=False)
    result = loader.wait_ready(timeout=0.1)
    assert not result


def test_get_model_returns_after_load(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(_FakeModel(), handle)
    manifest_path = _write_fake_manifest(tmp_path, str(model_path))

    loader = AsyncModelLoader(manifest_path, preload=True)
    loader.wait_ready(timeout=10.0)
    model = loader.get_model(timeout=1.0)
    assert model is not None
    proba = model.predict_proba([[0.1, 0.2]])
    assert proba[0][1] == 0.8


def test_reload_functionality(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(_FakeModel(), handle)
    manifest_path = _write_fake_manifest(tmp_path, str(model_path))

    loader = AsyncModelLoader(manifest_path, preload=True)
    loader.wait_ready(timeout=10.0)
    assert loader.is_ready()

    loader.reload()
    assert not loader.is_ready()
    loader.wait_ready(timeout=10.0)
    assert loader.is_ready()


def test_graceful_missing_manifest(tmp_path: Path) -> None:
    loader = AsyncModelLoader(tmp_path / "missing.json", preload=True)
    loader.wait_ready(timeout=10.0)
    assert not loader.is_ready()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        loader.get_model(timeout=1.0)
