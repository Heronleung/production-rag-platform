"""Validated schemas shared by the evaluation CLI, reports, and tests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Strategy = Literal["vanilla", "mmr", "multi_query", "combined"]


class GoldenSample(BaseModel):
    """One human-reviewed evaluation question and its expected evidence."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_sources: list[str] = Field(default_factory=list)
    reference_chunk_keys: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "GoldenSample":
        if len(self.reference_sources) != len(set(self.reference_sources)):
            raise ValueError("reference_sources contains duplicates")
        if len(self.reference_chunk_keys) != len(set(self.reference_chunk_keys)):
            raise ValueError("reference_chunk_keys contains duplicates")
        invalid = [key for key in self.reference_chunk_keys if "::" not in key]
        if invalid:
            raise ValueError(f"invalid chunk key(s), expected source::index: {invalid}")
        return self


class RetrievalScores(BaseModel):
    hit_rate_at_k: float | None = None
    mrr_at_k: float | None = None
    source_recall_at_k: float | None = None
    key_precision_at_k: float | None = None


class EvaluationResult(BaseModel):
    id: str
    question: str
    reference_answer: str
    answer: str
    retrieved_keys: list[str]
    retrieved_sources: list[str]
    retrieved_contexts: list[str]
    scores: RetrievalScores
    ragas_scores: dict[str, float | None] = Field(default_factory=dict)
    latency_seconds: float = Field(ge=0.0)
    error: str | None = None


class EvaluationReport(BaseModel):
    schema_version: int = 1
    created_at: str
    strategy: Strategy
    dataset_path: str
    dataset_sha256: str
    top_k: int
    embedding_model: str
    llm_model: str
    results: list[EvaluationResult]
    aggregate: dict[str, float | int | None]
    metadata: dict[str, Any] = Field(default_factory=dict)
