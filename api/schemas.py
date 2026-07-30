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

    # Phase 3: retrieval quality flags.
    # All default to off so existing clients need no changes.

    use_mmr: bool = Field(
        default=False,
        description=(
            "Apply Maximal Marginal Relevance to the retrieved candidates before "
            "building context. Reduces redundancy when multiple chunks say the same thing."
        ),
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "MMR relevance/diversity trade-off. 1.0 = pure relevance (degenerates to "
            "top-k). 0.0 = maximum diversity. Only used when use_mmr=true."
        ),
    )
    mmr_fetch_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many candidates to fetch before MMR selection. Defaults to top_k * 4. "
            "Only used when use_mmr=true."
        ),
    )
    multi_query: bool = Field(
        default=False,
        description=(
            "Generate alternative phrasings of the question using the LLM, embed and "
            "search each one, then merge and deduplicate the results. Improves recall "
            "when the corpus uses different terminology than the question."
        ),
    )
    multi_query_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of LLM-generated query variants. Only used when multi_query=true.",
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
