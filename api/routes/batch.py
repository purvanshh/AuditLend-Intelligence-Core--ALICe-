from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_auth
from services.metrics import batch_prediction_duration_seconds

logger = structlog.get_logger()

router = APIRouter()

_ML_AVAILABLE: bool | None = None


def _check_ml_available() -> bool:
    global _ML_AVAILABLE
    if _ML_AVAILABLE is not None:
        return _ML_AVAILABLE
    try:
        import xgboost
        _ML_AVAILABLE = True
    except ImportError:
        _ML_AVAILABLE = False
    return _ML_AVAILABLE


class BatchPredictRequest(BaseModel):
    features: list[dict[str, Any]]
    model_version: str = "XGB_V1"


class BatchPredictResponse(BaseModel):
    results: list[dict[str, Any]]
    batch_size: int
    latency_ms: float


class BatchStatusResponse(BaseModel):
    ml_available: bool
    cache_stats: dict[str, int] | None = None


@router.post("/batch/predict", dependencies=[Depends(require_auth)])
async def batch_predict(request: BatchPredictRequest) -> dict[str, Any]:
    if not _check_ml_available():
        raise HTTPException(
            status_code=424,
            detail="ML model not available: xgboost is not installed",
        )

    features = request.features
    if not features:
        return {"results": [], "batch_size": 0, "latency_ms": 0.0}

    max_total = 1000
    batch_limit = 50

    if len(features) > max_total:
        features = features[:max_total]

    started = time.perf_counter()
    all_results: list[dict[str, Any]] = []

    for i in range(0, len(features), batch_limit):
        batch = features[i:i + batch_limit]
        batch_results = _predict_batch(batch, request.model_version)
        all_results.extend(batch_results)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    batch_prediction_duration_seconds.labels(batch_size=len(features)).observe(latency_ms / 1000.0)

    return {
        "results": all_results,
        "batch_size": len(features),
        "latency_ms": latency_ms,
    }


@router.get("/batch/status", dependencies=[Depends(require_auth)])
async def batch_status() -> BatchStatusResponse:
    from ml.optimize.prediction_cache import PredictionCache
    cache = PredictionCache()
    return BatchStatusResponse(
        ml_available=_check_ml_available(),
        cache_stats=cache.stats(),
    )


def _predict_batch(batch: list[dict[str, Any]], model_version: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for feature_row in batch:
        results.append(
            {
                "prediction": 0.5,
                "model_version": model_version,
                "features": feature_row,
            }
        )
    return results
