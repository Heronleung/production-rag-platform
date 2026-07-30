"""FastAPI application entry point.

Run it with::

    uv run uvicorn api.main:app --reload

OpenAPI docs are served at ``/docs``.

The app owns three cross-cutting concerns and nothing else; all endpoint logic
lives in :mod:`api.routers`.

1. Structured logging, configured before anything else so startup is captured.
2. A request id middleware, which accepts an inbound ``X-Request-ID`` (so a
   gateway's id is preserved) or generates one, exposes it to the logging
   context, and echoes it back on the response.
3. A JSON error shape that always includes the request id, so a user-reported
   failure can be found in the logs.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import settings
from api.logging_config import configure_logging, request_id_var
from api.routers import health, ingest, query

configure_logging()
logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dependencies are built lazily on first use rather than here: the API must
    # still start (and report through /readyz) when Milvus or Ollama is down.
    logger.info(
        "api starting",
        extra={
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "milvus_collection": settings.milvus_collection,
        },
    )
    yield
    logger.info("api stopping")


app = FastAPI(
    title="Production RAG Platform API",
    version="0.2.0",
    description=(
        "Ingestion and retrieval-augmented generation over a Milvus-backed corpus. "
        "Embeddings and generation run on local Ollama models by default."
    ),
    lifespan=lifespan,
)

# Phase 5 adds a Next.js frontend on another origin; permissive in development
# only, to be tightened via configuration when the frontend lands.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable]
):
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request failed",
            extra={"method": request.method, "path": request.url.path},
        )
        raise
    finally:
        request_id_var.reset(token)

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request handled",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed_ms, 2),
            "request_id": request_id,
        },
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id_var.get()},
        headers={REQUEST_ID_HEADER: request_id_var.get()},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": request_id_var.get()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak internals to the client; the request id is the bridge to the log.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id_var.get()},
    )


app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {
        "service": "production-rag-platform",
        "version": app.version,
        "docs": "/docs",
    }
