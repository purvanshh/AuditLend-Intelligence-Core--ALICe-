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
        self._generation = 0
        self._load_in_progress = False

        if preload:
            self._start_loader()

    def _start_loader(self) -> None:
        with self._lock:
            if self._load_in_progress:
                return
            self._load_in_progress = True
        thread = threading.Thread(target=self._load_all, daemon=True)
        thread.start()

    def _ensure_loading(self) -> None:
        with self._lock:
            if self._ready.is_set() or self._load_in_progress:
                return
        self._start_loader()

    def _load_all(self) -> None:
        while True:
            with self._lock:
                generation = self._generation
            try:
                model = self.load_model()
                explainer = self.load_explainer(model)
            except Exception as exc:
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._load_error = exc
                    self._load_in_progress = False
                logger.error("async_model_load_failed", error=str(exc))
                self._ready.set()
                return
            with self._lock:
                if generation != self._generation:
                    continue
                self._model = model
                self._explainer = explainer
                self._load_error = None
                self._load_in_progress = False
            self._ready.set()
            logger.info("async_model_loaded", manifest_path=str(self._manifest_path))
            return

    def load_model(self) -> Any:
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self._manifest_path}")
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        artifact_path = Path(str(manifest.get("model_artifact_path", manifest.get("artifact_path", ""))))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        with artifact_path.open("rb") as handle:
            return pickle.load(handle)

    def load_explainer(self, model: Any = None) -> Any:
        try:
            import shap
        except ImportError:
            logger.warning("shap_not_available", message="SHAP not installed, explainer unavailable")
            return None
        if model is None:
            return None
        try:
            return shap.Explainer(model)
        except Exception as exc:
            logger.warning("shap_explainer_failed", error=str(exc))
            return None

    def wait_ready(self, timeout: float = 30.0) -> bool:
        self._ensure_loading()
        if not self._ready.wait(timeout=timeout):
            return False
        with self._lock:
            return self._load_error is None

    def is_ready(self) -> bool:
        return self._ready.is_set() and self._load_error is None

    def get_model(self, timeout: float = 30.0) -> Any:
        self._ensure_loading()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"Model not ready within {timeout}s")
        with self._lock:
            if self._load_error is not None:
                raise RuntimeError(f"Model load failed: {self._load_error}")
            if self._model is None:
                raise RuntimeError("Model was not loaded")
            return self._model

    def get_explainer(self, timeout: float = 30.0) -> Any:
        self._ensure_loading()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"Explainer not ready within {timeout}s")
        with self._lock:
            if self._load_error is not None:
                raise RuntimeError(f"Model load failed: {self._load_error}")
            if self._explainer is None:
                raise RuntimeError("Explainer was not loaded")
            return self._explainer

    def reload(self) -> None:
        with self._lock:
            self._generation += 1
            self._model = None
            self._explainer = None
            self._load_error = None
        self._ready.clear()
