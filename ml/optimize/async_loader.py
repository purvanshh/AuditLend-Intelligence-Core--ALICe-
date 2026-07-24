from __future__ import annotations

import json
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class AsyncModelLoader:
    def __init__(self, manifest_path: str | Path, preload: bool = True) -> None:
        self._manifest_path = Path(manifest_path)
        self._model: Any = None
        self._explainer: Any = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._load_error: Exception | None = None

        if preload:
            self._start_loader()

    def _start_loader(self) -> None:
        thread = threading.Thread(target=self._load_all, daemon=True)
        thread.start()

    def _load_all(self) -> None:
        try:
            model = self.load_model()
            explainer = self.load_explainer()
            with self._lock:
                self._model = model
                self._explainer = explainer
                self._load_error = None
            self._ready.set()
            logger.info("async_model_loaded", manifest_path=str(self._manifest_path))
        except Exception as exc:
            with self._lock:
                self._load_error = exc
            logger.error("async_model_load_failed", error=str(exc))
            self._ready.set()

    def load_model(self) -> Any:
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self._manifest_path}")
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        artifact_path = Path(str(manifest.get("model_artifact_path", manifest.get("artifact_path", ""))))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        with artifact_path.open("rb") as handle:
            return pickle.load(handle)

    def load_explainer(self) -> Any:
        try:
            import shap
            if self._model is None:
                return None
            return shap.Explainer(self._model)
        except ImportError:
            logger.warning("shap_not_available", message="SHAP not installed, explainer unavailable")
            return None

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout=timeout)

    def is_ready(self) -> bool:
        return self._ready.is_set() and self._load_error is None

    def get_model(self, timeout: float = 30.0) -> Any:
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"Model not ready within {timeout}s")
        with self._lock:
            if self._load_error is not None:
                raise RuntimeError(f"Model load failed: {self._load_error}")
            if self._model is None:
                raise RuntimeError("Model was not loaded")
            return self._model

    def get_explainer(self, timeout: float = 30.0) -> Any:
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"Explainer not ready within {timeout}s")
        with self._lock:
            if self._load_error is not None:
                raise RuntimeError(f"Model load failed: {self._load_error}")
            if self._explainer is None:
                raise RuntimeError("Explainer was not loaded")
            return self._explainer

    def reload(self) -> None:
        self._ready.clear()
        with self._lock:
            self._model = None
            self._explainer = None
            self._load_error = None
        self._start_loader()
