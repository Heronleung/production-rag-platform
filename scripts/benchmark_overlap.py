"""Phase 1 acceptance check: does Milvus return the same results as Chroma?

For every query it compares the top-k result sets of both backends and measures
latency. The acceptance criterion for Phase 1 is a mean top-5 overlap of at
least 80 percent; anything lower means the index parameters or the metric type
diverge and must be fixed before Phase 2 is started.

Usage:
    uv run python scripts/benchmark_overlap.py \\
        --queries evaluation/queries/smoke_queries.json \\
        --top-k 5 --out docs/benchmark.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from api.embeddings import OpenAIEmbedder
from api.vectorstore.base import VectorStore
from api.vectorstore.chroma_store import ChromaStore
from api.vectorstore.milvus_store import MilvusStore

ACCEPTANCE_THRESHOLD = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Chroma and Milvus retrieval.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark.md"))
    return parser.parse_args()


def timed_search(store: VectorStore, vector: list[float], top_k: int) -> tuple[set[str], float]:
    started = time.perf_counter()
    hits = store.search(vector, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {hit.key for hit in hits}, elapsed_ms


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    args = parse_args()
    queries: list[str] = json.loads(args.queries.read_text(encoding="utf-8"))

    embedder = OpenAIEmbedder()
    vectors = embedder.embed(queries)

    chroma = ChromaStore()
    milvus = MilvusStore()

    overlaps: list[float] = []
    chroma_latencies: list[float] = []
    milvus_latencies: list[float] = []

    for query, vector in zip(queries, vectors, strict=True):
        chroma_keys, chroma_ms = timed_search(chroma, vector, args.top_k)
        milvus_keys, milvus_ms = timed_search(milvus, vector, args.top_k)

        denominator = max(len(chroma_keys), 1)
        overlap = len(chroma_keys & milvus_keys) / denominator

        overlaps.append(overlap)
        chroma_latencies.append(chroma_ms)
        milvus_latencies.append(milvus_ms)
        print(f"{overlap:6.0%}  {milvus_ms:7.1f} ms  {query[:60]}")

    mean_overlap = statistics.fmean(overlaps)
    report = f"""# Phase 1 benchmark: ChromaDB vs Milvus

Queries: {len(queries)} | top_k: {args.top_k} | metric: {milvus.metric_type}
HNSW: M={milvus.hnsw_m}, efConstruction={milvus.ef_construction}, ef={milvus.ef_search}

| Metric | ChromaDB | Milvus |
| --- | --- | --- |
| Rows stored | {chroma.count()} | {milvus.count()} |
| Mean latency (ms) | {statistics.fmean(chroma_latencies):.1f} | {statistics.fmean(milvus_latencies):.1f} |
| P50 latency (ms) | {percentile(chroma_latencies, 0.50):.1f} | {percentile(milvus_latencies, 0.50):.1f} |
| P95 latency (ms) | {percentile(chroma_latencies, 0.95):.1f} | {percentile(milvus_latencies, 0.95):.1f} |

| Overlap metric | Value |
| --- | --- |
| Mean top-{args.top_k} overlap | {mean_overlap:.1%} |
| Worst query overlap | {min(overlaps):.1%} |
| Queries with full overlap | {sum(1 for value in overlaps if value == 1.0)}/{len(overlaps)} |

Acceptance threshold: {ACCEPTANCE_THRESHOLD:.0%} mean overlap -> \
**{"PASS" if mean_overlap >= ACCEPTANCE_THRESHOLD else "FAIL"}**
"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nMean top-{args.top_k} overlap: {mean_overlap:.1%}")
    print(f"Report written to {args.out}")

    return 0 if mean_overlap >= ACCEPTANCE_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
