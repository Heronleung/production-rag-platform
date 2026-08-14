"""Pydantic request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.config import LLMProvider


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
        description="When true the answer is streamed as SSE; otherwise JSON is returned.",
    )
    llm_provider: LLMProvider | None = Field(
        default=None,
        description="Optional per-request chat provider. Defaults to server configuration.",
    )
    llm_model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional per-request chat model. API keys remain server-side.",
    )
    use_mmr: bool = Field(
        default=False,
        description="Apply Maximal Marginal Relevance to reduce redundant passages.",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR relevance/diversity trade-off.",
    )
    mmr_fetch_k: int | None = Field(
        default=None,
        ge=1,
        description="Candidates to fetch before MMR selection. Defaults to top_k * 4.",
    )
    multi_query: bool = Field(
        default=False,
        description="Generate alternative phrasings and merge their retrieval results.",
    )
    multi_query_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of generated query variants.",
    )


class LLMCheckRequest(BaseModel):
    provider: LLMProvider
    model: str | None = Field(default=None, min_length=1, max_length=200)


class LLMCheckResponse(BaseModel):
    ok: bool
    provider: LLMProvider
    model: str
    detail: str


class Citation(BaseModel):
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
