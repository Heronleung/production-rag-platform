"""``POST /ingest`` - upload a document, chunk it, embed it and store it.

The pipeline is deliberately the same one the Phase 1 scripts use
(:mod:`ingestion.chunking` plus the configured embedder and vector store), so a
document ingested over HTTP is indistinguishable from one ingested offline.

Embedding is CPU/network bound and blocking, so the handler is a plain ``def``:
FastAPI then runs it in a worker thread and the event loop stays free to serve
other requests. Making it ``async def`` and calling the blocking embedder inside
would stall every other connection.
"""

from __future__ import annotations

import logging
import time
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from api.dependencies import get_embedder_singleton, get_vector_store
from api.embeddings import Embedder
from api.schemas import IngestResponse
from api.vectorstore.base import Chunk, VectorStore
from ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SUPPORTED_SUFFIXES,
    decode_text_bytes,
    read_pdf_bytes,
    split_text,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
EMBED_BATCH_SIZE = 64


def _safe_source_name(filename: str | None) -> str:
    """Reduce an uploaded filename to a bare name.

    The value is stored as chunk metadata and echoed back to clients, so path
    components from the client are stripped rather than trusted. Windows-style
    separators are normalised first, since PurePosixPath does not treat them as
    separators.
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file has no filename."
        )
    name = PurePosixPath(filename.replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid filename: {filename!r}"
        )
    return name


def _extract_text(name: str, data: bytes) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix or name}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )
    if suffix == ".pdf":
        try:
            return read_pdf_bytes(data)
        except RuntimeError as exc:  # pypdf missing
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001 - malformed PDF is a client error
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read PDF: {exc}",
            ) from exc
    return decode_text_bytes(data)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Chunk, embed and store an uploaded document",
)
def ingest(
    file: UploadFile = File(..., description="A .pdf, .md or .txt document."),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=8000),
    chunk_overlap: int = Query(DEFAULT_CHUNK_OVERLAP, ge=0, le=2000),
    embedder: Embedder = Depends(get_embedder_singleton),
    store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    started = time.perf_counter()
    source = _safe_source_name(file.filename)
    data = file.file.read()

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )

    text = _extract_text(source, data)
    pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not pieces:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No extractable text found in '{source}'.",
        )

    written = 0
    for start in range(0, len(pieces), EMBED_BATCH_SIZE):
        window = pieces[start : start + EMBED_BATCH_SIZE]
        vectors = embedder.embed(window)
        written += store.upsert(
            [
                Chunk(text=piece, source=source, chunk_index=start + offset, vector=vector)
                for offset, (piece, vector) in enumerate(zip(window, vectors, strict=True))
            ]
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "ingest completed",
        extra={
            "source": source,
            "bytes": len(data),
            "chunks": written,
            "elapsed_seconds": round(elapsed, 3),
        },
    )
    return IngestResponse(
        source=source,
        chunks_written=written,
        embedding_model=embedder.describe(),
        elapsed_seconds=round(elapsed, 3),
    )
