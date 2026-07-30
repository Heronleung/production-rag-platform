"""Offline tests for Phase 4 dataset, metrics, aggregation and regression gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset import load_dataset
from evaluation.metrics import aggregate_results, score_retrieval
from evaluation.regression import compare_reports
from evaluation.schema import (
    EvaluationReport,
    EvaluationResult,
    GoldenSample,
    RetrievalScores,
)


def sample(**updates) -> GoldenSample:  # noqa: ANN003
    data = {
        "id": "q1",
        "question": "question",
        "reference_answer": "answer",
        "reference_sources": ["doc.md"],
        "reference_chunk_keys": ["doc.md::2"],
    }
    data.update(updates)
    return GoldenSample.model_validate(data)


def result(scores: RetrievalScores, latency: float = 0.1) -> EvaluationResult:
    return EvaluationResult(
        id="q1",
        question="q",
        reference_answer="a",
        answer="a",
        retrieved_keys=["doc.md::2"],
        retrieved_sources=["doc.md"],
        retrieved_contexts=["context"],
        scores=scores,
        latency_seconds=latency,
    )


def report(**aggregate) -> EvaluationReport:  # noqa: ANN003
    return EvaluationReport(
        created_at="2026-07-30T00:00:00+00:00",
        strategy="vanilla",
        dataset_path="golden.jsonl",
        dataset_sha256="abc",
        top_k=5,
        embedding_model="fake:embed",
        llm_model="fake:chat",
        results=[],
        aggregate=aggregate,
    )


def test_golden_sample_rejects_invalid_chunk_key() -> None:
    with pytest.raises(ValueError, match="source::index"):
        sample(reference_chunk_keys=["missing-separator"])


def test_golden_sample_rejects_duplicate_references() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        sample(reference_sources=["doc.md", "doc.md"])


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    line = json.dumps(sample().model_dump())
    path.write_text(f"{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate sample id"):
        load_dataset(path)


def test_load_dataset_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok", "question": "q", "reference_answer": "a"}\nnot-json\n')
    with pytest.raises(ValueError, match=":2:"):
        load_dataset(path)


def test_hit_rate_and_mrr_at_first_position() -> None:
    scores = score_retrieval(
        ["doc.md::2", "doc.md::9"], ["doc.md", "doc.md"], sample()
    )
    assert scores.hit_rate_at_k == 1.0
    assert scores.mrr_at_k == 1.0
    assert scores.key_precision_at_k == 0.5
    assert scores.source_recall_at_k == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    scores = score_retrieval(
        ["x::1", "x::2", "doc.md::2"], ["x", "x", "doc.md"], sample()
    )
    assert scores.hit_rate_at_k == 1.0
    assert scores.mrr_at_k == pytest.approx(1 / 3)


def test_missing_retrieval_annotations_are_none_not_zero() -> None:
    scores = score_retrieval(
        ["x::1"], ["x"], sample(reference_sources=[], reference_chunk_keys=[])
    )
    assert scores.hit_rate_at_k is None
    assert scores.mrr_at_k is None
    assert scores.key_precision_at_k is None
    assert scores.source_recall_at_k is None


def test_source_recall_supports_multiple_expected_sources() -> None:
    scores = score_retrieval(
        [], ["one.md"], sample(reference_sources=["one.md", "two.md"], reference_chunk_keys=[])
    )
    assert scores.source_recall_at_k == 0.5


def test_aggregate_ignores_none_metrics() -> None:
    aggregate = aggregate_results([
        result(RetrievalScores(hit_rate_at_k=1.0), latency=0.1),
        result(RetrievalScores(hit_rate_at_k=None), latency=0.4),
    ])
    assert aggregate["hit_rate_at_k"] == 1.0
    assert aggregate["sample_count"] == 2
    assert aggregate["latency_p95_seconds"] == 0.4


def test_regression_gate_passes_within_tolerance() -> None:
    baseline = report(hit_rate_at_k=1.0, mrr_at_k=0.8)
    current = report(hit_rate_at_k=1.0, mrr_at_k=0.785)
    assert compare_reports(baseline, current) == []


def test_regression_gate_fails_beyond_tolerance() -> None:
    baseline = report(hit_rate_at_k=1.0, mrr_at_k=0.8)
    current = report(hit_rate_at_k=0.9, mrr_at_k=0.7)
    failures = compare_reports(baseline, current)
    assert {failure.metric for failure in failures} == {"hit_rate_at_k", "mrr_at_k"}


def test_regression_gate_skips_unannotated_metrics() -> None:
    baseline = report(hit_rate_at_k=None)
    current = report(hit_rate_at_k=None)
    assert compare_reports(baseline, current) == []
