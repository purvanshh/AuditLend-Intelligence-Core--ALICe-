from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.auth import require_auth
from ml.monitoring.drift_reporter import EVIDENTLY_AVAILABLE, EvidentlyDriftReporter

router = APIRouter()
logger = structlog.get_logger()

REPORTS_DIR = Path("monitoring/reports")


class DriftRequest(BaseModel):
    reference_data: list[dict[str, Any]]
    candidate_data: list[dict[str, Any]]
    drift_share_threshold: float = 0.1


class ReportGenerateRequest(BaseModel):
    feature_names: list[str] | None = None
    p_value_threshold: float | None = None


@router.post(
    "/monitoring/drift",
    dependencies=[Depends(require_auth)],
)
async def check_drift(request: DriftRequest) -> JSONResponse:
    reference_df = pd.DataFrame(request.reference_data)
    candidate_df = pd.DataFrame(request.candidate_data)

    reporter = EvidentlyDriftReporter(reference_df)
    result = reporter.compute_data_drift(candidate_df, request.drift_share_threshold)

    logger.info(
        "drift_check",
        ref_rows=len(reference_df),
        cand_rows=len(candidate_df),
        drifted_features=len(result["drifted_features"]),
        drift_share=round(result["drift_share"], 4),
        step="DRIFT_CHECK",
    )

    return JSONResponse(content=result)


@router.get("/monitoring/reports")
async def list_reports() -> JSONResponse:
    reports_dir = REPORTS_DIR
    if not reports_dir.exists():
        return JSONResponse(content={"reports": []})

    entries = []
    for child in sorted(reports_dir.iterdir()):
        if child.is_file():
            entries.append({
                "name": child.name,
                "path": str(child),
                "size_bytes": child.stat().st_size,
            })

    return JSONResponse(content={"reports": entries})


@router.post(
    "/monitoring/reports/generate",
    dependencies=[Depends(require_auth)],
)
async def generate_report(request: ReportGenerateRequest) -> JSONResponse:
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "drift_report.html"
    suite_path = reports_dir / "test_suite.html"

    reference_df = pd.DataFrame()
    candidate_df = pd.DataFrame()

    reporter = EvidentlyDriftReporter(reference_df)

    html_path = reporter.generate_html_report(candidate_df, report_path)
    suite_result = reporter.generate_test_suite(candidate_df, suite_path)

    return JSONResponse(content={
        "report_path": str(html_path),
        "test_suite_path": str(suite_result),
        "evidently_available": EVIDENTLY_AVAILABLE,
    })
