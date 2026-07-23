from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import require_auth
from api.dependencies import get_async_session
from services.document_parser import (
    extract_bank_statement_features,
    parse_document_bytes,
)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "text/plain",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/apply-loan/documents",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_auth)],
)
async def upload_document(
    file: UploadFile = File(...),
    application_id: str | None = None,
    document_type: str | None = None,
    session: Annotated[AsyncSession, Depends(get_async_session)] = None,
    auth: Annotated[None, Depends(require_auth)] = None,
) -> dict[str, Any]:
    """Upload and parse a financial document.

    Accepts PDF, PNG, JPG files. Returns extracted features.
    The document is not stored — only parsed features are returned.

    Args:
        file: The uploaded document file.
        application_id: Optional application ID to associate with the document.
        document_type: Optional hint ("bank_statement", "salary_slip", "gst_filing").
        session: Database session (injected).
        auth: Auth dependency (injected).

    Returns:
        Dict with filename, document_type, features, confidence, and warnings.
    """
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}. "
                   f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    filename = file.filename or "upload"
    features = parse_document_bytes(content, filename, document_type)

    return {
        "filename": filename,
        "document_type": features.document_type,
        "features": extract_bank_statement_features(features),
        "confidence": features.confidence,
        "warnings": features.errors[:3],
    }
