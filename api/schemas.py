"""Pydantic request/response models for the HTTP API.

Keeping these out of the router functions is what makes the auto-generated
OpenAPI docs at ``/docs`` useful: every field carries a type, a constraint and,
where the meaning is not obvious, a description.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    source: str = Field(description="Name the chunks were stored under.")
    chunks_written: int = Field(description="Number of chunks written to the vector store.")
    embedding_model: str = Field(description="Provider and model used to embed the chunks.")
    elapsed_seconds: float = Field(description="Server-side processing time.")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's natural-language question.")
    top_k: int = Field(5, ge=1, le=50, description="How many chunks to retrieve as context.")
    source_filter: str | None = Field(
        default=None,
        description="Restrict retrieval to chunks whose source matches this value exactly.",
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature.")
    stream: bool = Field(
        default=True,
        description="When true the answer is streamed as SSE; when false a single JSON body "
        "is returned.",
    )


class Citation(BaseModel):
    """One retrieved chunk, returned alongside the answer so the client can
    show where each claim came from."""

    source: str
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    llm_model: str
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the process is alive.")


class ReadinessDependency(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    status: str = Field(description="'ready' only when every dependency is reachable.")
    dependencies: list[ReadinessDependency]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
