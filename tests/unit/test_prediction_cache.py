from __future__ import annotations

import threading
import time
from pathlib import Path

from ml.optimize.prediction_cache import (
    PredictionCache,
    cache_key_for_features,
    cached_explain,
)


def test_cache_set_and_get() -> None:
    cache = PredictionCache(max_size=100, ttl_seconds=3600)
    cache.set("key1", {"prediction": 0.8})
    result = cache.get("key1")
    assert result == {"prediction": 0.8}


def test_cache_miss_returns_none() -> None:
    cache = PredictionCache(max_size=100, ttl_seconds=3600)
    result = cache.get("nonexistent")
    assert result is None


def test_cache_ttl_expiry() -> None:
    cache = PredictionCache(max_size=100, ttl_seconds=1)
    cache.set("key1", {"prediction": 0.8})
    assert cache.get("key1") is not None
    time.sleep(1.1)
    assert cache.get("key1") is None


def test_cache_invalidate() -> None:
    cache = PredictionCache(max_size=100, ttl_seconds=3600)
    cache.set("apple:1", {"v": 1})
    cache.set("apple:2", {"v": 2})
    cache.set("banana:1", {"v": 3})
    removed = cache.invalidate("apple")
    assert removed == 2
    assert cache.get("apple:1") is None
    assert cache.get("apple:2") is None
    assert cache.get("banana:1") is not None


def test_cache_max_size_eviction() -> None:
    cache = PredictionCache(max_size=3, ttl_seconds=3600)
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.set("c", {"v": 3})
    cache.set("d", {"v": 4})
    assert cache.get("a") is None
    assert cache.get("d") is not None


def test_cache_stats() -> None:
    cache = PredictionCache(max_size=100, ttl_seconds=3600)
    cache.get("miss1")
    cache.get("miss2")
    cache.set("hit1", {"v": 1})
    cache.get("hit1")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 1
    assert stats["max_size"] == 100


def test_thread_safety() -> None:
    cache = PredictionCache(max_size=1000, ttl_seconds=3600)

    def _writer():
        for i in range(100):
            cache.set(f"k{i}", {"v": i})

    def _reader():
        for i in range(100):
            cache.get(f"k{i}")

    threads = [threading.Thread(target=_writer) for _ in range(4)]
    threads += [threading.Thread(target=_reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = cache.stats()
    assert stats["size"] > 0


def test_cache_key_determinism() -> None:
    features = {"a": 1, "b": 2, "c": "test"}
    key1 = cache_key_for_features(features, "XGB_V1")
    key2 = cache_key_for_features(features, "XGB_V1")
    key3 = cache_key_for_features({"c": "test", "a": 1, "b": 2}, "XGB_V1")
    assert key1 == key2
    assert key1 == key3


def test_cache_key_differs_for_different_versions() -> None:
    features = {"a": 1, "b": 2}
    key_v1 = cache_key_for_features(features, "XGB_V1")
    key_v2 = cache_key_for_features(features, "XGB_V2")
    assert key_v1 != key_v2


def test_cached_explain_decorator() -> None:
    cache = PredictionCache(max_size=100, ttl_seconds=3600)
    call_count = 0

    def _fake_explain(feature_row, **kwargs):
        nonlocal call_count
        call_count += 1
        class FakeResult:
            def to_audit_payload(self):
                return {"prediction": 0.8, "model_version": "XGB_V1"}
        return FakeResult()

    decorated = cached_explain(cache)(_fake_explain)

    result1 = decorated({"a": 1}, model_version="XGB_V1")
    result2 = decorated({"a": 1}, model_version="XGB_V1")
    assert call_count == 1
    assert result1.to_audit_payload() == {"prediction": 0.8, "model_version": "XGB_V1"}
