"""Deterministic retrieval metrics and report aggregation.

Unlike LLM-judged metrics, these functions have no model calls and are suitable
for a hard CI regression gate.
"""

from __future__ import annotations

from statistics import mean

from evaluation.schema import EvaluationResult, GoldenSample, RetrievalScores


def score_retrieval(
    retrieved_keys: list[str], retrieved_sources: list[str], sample: GoldenSample
) -> RetrievalScores:
    """Score retrieved evidence against the annotations available on a sample.

    Metrics whose required annotation is absent are ``None`` rather than zero;
    this prevents a partially annotated dataset from reporting fake failures.
    """
    expected_keys = set(sample.reference_chunk_keys)
    expected_sources = set(sample.reference_sources)

    hit_rate: float | None = None
    mrr: float | None = None
    key_precision: float | None = None
    if expected_keys:
        relevant_positions = [
            rank for rank, key in enumerate(retrieved_keys, start=1) if key in expected_keys
        ]
        hit_rate = 1.0 if relevant_positions else 0.0
        mrr = 1.0 / min(relevant_positions) if relevant_positions else 0.0
        key_precision = (
            sum(key in expected_keys for key in retrieved_keys) / len(retrieved_keys)
            if retrieved_keys
            else 0.0
        )

    source_recall: float | None = None
    if expected_sources:
        source_recall = len(expected_sources.intersection(retrieved_sources)) / len(expected_sources)

    return RetrievalScores(
        hit_rate_at_k=hit_rate,
        mrr_at_k=mrr,
        source_recall_at_k=source_recall,
        key_precision_at_k=key_precision,
    )


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(mean(present), 6) if present else None


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.999999)) - 1))
    return ordered[index]


def aggregate_results(results: list[EvaluationResult]) -> dict[str, float | int | None]:
    """Aggregate per-sample metrics while ignoring unannotated ``None`` values."""
    latencies = [result.latency_seconds for result in results]
    return {
        "sample_count": len(results),
        "error_count": sum(result.error is not None for result in results),
        "hit_rate_at_k": _mean_present([r.scores.hit_rate_at_k for r in results]),
        "mrr_at_k": _mean_present([r.scores.mrr_at_k for r in results]),
        "source_recall_at_k": _mean_present([r.scores.source_recall_at_k for r in results]),
        "key_precision_at_k": _mean_present([r.scores.key_precision_at_k for r in results]),
        "latency_mean_seconds": round(mean(latencies), 6) if latencies else 0.0,
        "latency_p50_seconds": round(_nearest_rank(latencies, 0.50), 6),
        "latency_p95_seconds": round(_nearest_rank(latencies, 0.95), 6),
    }
