# Production RAG Platform

A production-grade Retrieval-Augmented Generation platform: **Milvus** vector store, **FastAPI**
backend, **RAGAS** evaluation, **Kubernetes** deployment, **GitHub Actions** CI/CD and
**Prometheus + Grafana** observability.

This repository is built phase by phase. See `docs/roadmap.md` for the full plan.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Project skeleton, tooling, lint/test setup | in progress |
| 1 | Replace ChromaDB with Milvus behind a `VectorStore` interface | in progress |
| 2 | FastAPI backend (`/ingest`, `/query`, SSE streaming) | not started |
| 3 | Retrieval quality (chunking, MMR, multi-query, reranking) | not started |
| 4 | RAGAS evaluation pipeline + regression gate | not started |
| 5 | Next.js frontend | not started |
| 6 | Docker + Kubernetes manifests | not started |
| 7 | CI/CD pipeline | not started |
| 8 | Monitoring + documentation | not started |

---

## Phase 1 quick start

### 1. Install dependencies

```bash
uv sync --extra dev
cp .env.example .env    # then fill in OPENAI_API_KEY
```

### 2. Start Milvus standalone

Milvus standalone is not a single container: it needs `etcd` for metadata and `MinIO` for object
storage.

```bash
docker compose -f deploy/compose/milvus.yml up -d
curl -f http://localhost:9091/healthz    # expect: OK
```

### 3. Migrate existing ChromaDB embeddings into Milvus

The migration reuses the embeddings already stored in Chroma, so no re-embedding cost is incurred.

```bash
uv run python scripts/migrate_chroma_to_milvus.py --batch-size 1000
```

### 4. Verify parity between the two backends

```bash
uv run python scripts/benchmark_overlap.py \
  --queries evaluation/queries/smoke_queries.json \
  --top-k 5 \
  --out docs/benchmark.md
```

Acceptance criterion: **top-5 overlap between Chroma and Milvus >= 80%**.

### 5. Run the test suite

```bash
uv run ruff check .
uv run pytest -m "not integration"        # contract tests, no services required
uv run pytest -m integration              # requires a running Milvus
```

---

## Architecture note: why an interface?

All business logic depends on `api.vectorstore.base.VectorStore`, never on a concrete client.
That makes the backend swap a configuration change instead of a rewrite, and it lets the same
contract test suite run against every implementation:

```
api/vectorstore/
├── base.py           # Chunk, SearchHit, VectorStore (ABC)
├── memory_store.py   # pure-Python reference implementation, used in unit tests
├── chroma_store.py   # legacy backend, kept for migration + parity benchmarking
└── milvus_store.py   # production backend, HNSW index
```

## Why Milvus over ChromaDB

| Concern | ChromaDB | Milvus |
| --- | --- | --- |
| Horizontal scale | single node | sharding + replication |
| Index options | HNSW only | HNSW, IVF_FLAT, IVF_PQ, DiskANN, SCANN |
| Filtered search | basic metadata filter | scalar field expressions, partition keys |
| Operations | embedded library | standalone/distributed service, K8s ready |

The trade-off is operational complexity: Milvus requires etcd and MinIO, persistent volumes, and
an explicit `load_collection()` step before any search can be served.
