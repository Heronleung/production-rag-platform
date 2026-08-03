"""Liveness and readiness probes.

The split matters once this runs on Kubernetes (Phase 6):

* ``/healthz`` answers "is the process alive?" - never touches a dependency, so
  a slow Milvus can never cause a restart loop.
* ``/readyz`` answers "can this pod serve traffic?" - checks the embedding
  provider, configured chat model and vector store, returning 503 when any are
  unavailable so the pod is removed from Service endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, Request, Response, status

from api.config import settings
from api.dependencies import get_chat_model, get_embedder_singleton, get_vector_store
from api.schemas import HealthResponse, ReadinessDependency, ReadinessResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])
T = TypeVar("T")


def _resolve_dependency(request: Request, provider: Callable[[], T]) -> T:
    """Honor FastAPI overrides while keeping probe failures catchable here."""
    resolver = request.app.dependency_overrides.get(provider, provider)
    return resolver()


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse)
def readyz(request: Request, response: Response) -> ReadinessResponse:
    checks: list[ReadinessDependency] = []

    try:
        embedder = _resolve_dependency(request, get_embedder_singleton)
        checks.append(
            ReadinessDependency(name="embedder", ok=True, detail=embedder.describe())
        )
    except Exception as exc:  # noqa: BLE001 - probe must not raise
        logger.warning("readiness check failed", extra={"dependency": "embedder"})
        checks.append(ReadinessDependency(name="embedder", ok=False, detail=str(exc)))

    try:
        chat_model = _resolve_dependency(request, get_chat_model)
        checks.append(
            ReadinessDependency(
                name="chat_model",
                ok=True,
                detail=chat_model.check_ready(),
            )
        )
    except Exception as exc:  # noqa: BLE001 - probe must not raise
        logger.warning("readiness check failed", extra={"dependency": "chat_model"})
        checks.append(ReadinessDependency(name="chat_model", ok=False, detail=str(exc)))

    try:
        store = _resolve_dependency(request, get_vector_store)
        checks.append(
            ReadinessDependency(
                name="vector_store",
                ok=True,
                detail=f"{settings.milvus_collection}: {store.count()} chunk(s)",
            )
        )
    except Exception as exc:  # noqa: BLE001 - probe must not raise
        logger.warning("readiness check failed", extra={"dependency": "vector_store"})
        checks.append(ReadinessDependency(name="vector_store", ok=False, detail=str(exc)))

    ready = all(check.ok for check in checks)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "not_ready", dependencies=checks)
