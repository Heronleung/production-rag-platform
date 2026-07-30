"""Run the production retrieval and answer pipeline over a golden dataset."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from api.embeddings import Embedder
from api.llm import ChatModel
from api.retrieval.pipeline import retrieve
from api.routers.query import build_messages
from api.vectorstore.base import VectorStore
from evaluation.metrics import aggregate_results, score_retrieval
from evaluation.schema import (
    EvaluationReport,
    EvaluationResult,
    GoldenSample,
    RetrievalScores,
    Strategy,
)


def _strategy_options(strategy: Strategy) -> dict[str, bool]:
    return {
        "use_mmr": strategy in {"mmr", "combined"},
        "multi_query": strategy in {"multi_query", "combined"},
    }


def run_evaluation(
    samples: list[GoldenSample],
    *,
    strategy: Strategy,
    dataset_path: str,
    dataset_hash: str,
    embedder: Embedder,
    store: VectorStore,
    llm: ChatModel,
    top_k: int = 5,
    mmr_lambda: float = 0.5,
    multi_query_count: int = 3,
) -> EvaluationReport:
    """Evaluate every sample; an individual failure is recorded, not fatal."""
    options = _strategy_options(strategy)
    results: list[EvaluationResult] = []

    for sample in samples:
        started = time.perf_counter()
        try:
            hits = retrieve(
                query=sample.question,
                embedder=embedder,
                store=store,
                llm=llm,
                top_k=top_k,
                use_mmr=options["use_mmr"],
                mmr_lambda=mmr_lambda,
                multi_query=options["multi_query"],
                multi_query_count=multi_query_count,
            )
            answer = llm.complete(build_messages(sample.question, hits), temperature=0.0)
            keys = [hit.key for hit in hits]
            sources = [hit.source for hit in hits]
            contexts = [hit.text for hit in hits]
            scores = score_retrieval(keys, sources, sample)
            error = None
        except Exception as exc:  # noqa: BLE001 - preserve the rest of an eval run
            answer = ""
            keys = []
            sources = []
            contexts = []
            scores = RetrievalScores()
            error = f"{type(exc).__name__}: {exc}"

        results.append(
            EvaluationResult(
                id=sample.id,
                question=sample.question,
                reference_answer=sample.reference_answer,
                answer=answer,
                retrieved_keys=keys,
                retrieved_sources=sources,
                retrieved_contexts=contexts,
                scores=scores,
                latency_seconds=round(time.perf_counter() - started, 6),
                error=error,
            )
        )

    return EvaluationReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        strategy=strategy,
        dataset_path=dataset_path,
        dataset_sha256=dataset_hash,
        top_k=top_k,
        embedding_model=embedder.describe(),
        llm_model=llm.describe(),
        results=results,
        aggregate=aggregate_results(results),
        metadata={
            "mmr_lambda": mmr_lambda,
            "multi_query_count": multi_query_count,
        },
    )
