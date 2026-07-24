from __future__ import annotations

import functools
import os
import threading
import time
from collections import OrderedDict
from hashlib import sha256
from typing import Any, Callable

import structlog

from services.metrics import (
    prediction_cache_hits_total,
    prediction_cache_misses_total,
    prediction_cache_size,
)

logger = structlog.get_logger()


class PredictionCache:
    def __init__(self, max_size: int | None = None, ttl_seconds: int | None = None) -> None:
        self._max_size = max_size or int(os.environ.get("PREDICTION_CACHE_SIZE", "1024"))
        self._ttl_seconds = ttl_seconds or int(os.environ.get("PREDICTION_CACHE_TTL", "3600"))
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                prediction_cache_misses_total.inc()
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                prediction_cache_misses_total.inc()
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            prediction_cache_hits_total.inc()
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.monotonic(), value)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            prediction_cache_size.set(len(self._cache))

    def invalidate(self, pattern: str) -> int:
        removed = 0
        with self._lock:
            keys_to_delete = [k for k in self._cache if pattern in k]
            for k in keys_to_delete:
                del self._cache[k]
                removed += 1
            prediction_cache_size.set(len(self._cache))
        logger.info("cache_invalidated", pattern=pattern, removed=removed)
        return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
            }


def cache_key_for_features(feature_row: dict[str, Any], model_version: str) -> str:
    digest = sha256()
    for k in sorted(feature_row):
        digest.update(k.encode("utf-8"))
        digest.update(str(feature_row[k]).encode("utf-8"))
    digest.update(model_version.encode("utf-8"))
    return digest.hexdigest()


def cached_explain(cache: PredictionCache) -> Callable:
    def decorator(explain_fn: Callable) -> Callable:
        @functools.wraps(explain_fn)
        def wrapper(feature_row: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            model_version = kwargs.get("model_version", "XGB_V1")
            key = cache_key_for_features(feature_row, model_version)
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = explain_fn(feature_row, *args, **kwargs)
            cache.set(key, result.to_audit_payload() if hasattr(result, "to_audit_payload") else result)
            return result

        return wrapper

    return decorator
