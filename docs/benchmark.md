# Phase 1 benchmark: ChromaDB vs Milvus

> This file is overwritten by `scripts/benchmark_overlap.py`. The table below is
> a placeholder showing the expected shape of the report.

Queries: - | top_k: 5 | metric: COSINE
HNSW: M=16, efConstruction=200, ef=64

| Metric | ChromaDB | Milvus |
| --- | --- | --- |
| Rows stored | - | - |
| Mean latency (ms) | - | - |
| P50 latency (ms) | - | - |
| P95 latency (ms) | - | - |

| Overlap metric | Value |
| --- | --- |
| Mean top-5 overlap | - |
| Worst query overlap | - |
| Queries with full overlap | - |

Acceptance threshold: 80% mean overlap -> pending

## How to read this report

- **Overlap** answers "did we break retrieval by switching backend?". Anything
  below 80 percent usually means the metric type differs (cosine vs L2) or the
  index has not finished building.
- **Latency** is measured client side and includes network round trip. Chroma
  runs in-process, so it will look faster on a tiny local dataset. That gap
  reverses once the collection no longer fits in one process.
- Re-run the benchmark after every HNSW parameter change and keep the numbers
  in the commit message, so the tuning history stays auditable.

## Tuning notes

| Parameter | Effect | Cost of increasing |
| --- | --- | --- |
| `M` | more graph neighbours, higher recall | memory, build time |
| `efConstruction` | better graph quality at build time | build time only |
| `ef` | wider search at query time, higher recall | query latency |
