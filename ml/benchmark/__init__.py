"""Benchmarking utilities for heuristic versus ML comparisons."""

from __future__ import annotations

from typing import Any

__all__ = ["BenchmarkReport", "benchmark_manifest"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from ml.benchmark.heuristic_vs_ml import BenchmarkReport, benchmark_manifest

        exports = {
            "BenchmarkReport": BenchmarkReport,
            "benchmark_manifest": benchmark_manifest,
        }
        return exports[name]
    raise AttributeError(name)
