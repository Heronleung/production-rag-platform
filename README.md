# Production RAG Platform

A production-grade Retrieval-Augmented Generation platform: **Milvus** vector store, **FastAPI**
backend, **RAGAS** evaluation, **Kubernetes** deployment, **GitHub Actions** CI/CD and
**Prometheus + Grafana** observability.

The whole stack runs locally with **no API key**: embeddings and generation default to
[Ollama](https://ollama.com). OpenAI is supported as a drop-in alternative behind the same
interface.

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
cp .env.example .env
```

The defaults in `.env.example` are already local-only. Nothing needs to be filled in unless you
want to switch to OpenAI.

### 2. Start the local models

```bash
ollama serve
ollama pull nomic-embed-text     # embeddings, 768 dimensions, ~274MB
ollama pull qwen2.5:1.5b         # generation, ~1.0GB
```

Or run Ollama in a container instead:

```bash
docker compose -f deploy/compose/ollama.yml up -d
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull qwen2.5:1.5b
```

#### Choosing a chat model

| Model | Size | When to use it |
| --- | --- | --- |
| `qwen2.5:0.5b` | ~0.4GB | Smoke tests and CI only. Do not publish benchmark numbers from it. |
| `qwen2.5:1.5b` | ~1.0GB | **Default.** Runs on CPU, strong for its size, good Chinese support. |
| `llama3.2:3b` | ~2.0GB | Better instruction following and citation discipline. |
| `llama3.1:8b` | ~4.7GB | Best quality here, but wants a GPU or 16GB+ RAM. |

The chat model can be changed at any time - it never touches stored vectors. The **embedding**
model cannot: its dimension is baked into the Milvus collection schema, so changing it means
re-embedding everything and recreating the collection.

### 3. Verify the embedder before touching Milvus

```bash
uv run python scripts/check_embedder.py
```

This prints the provider, model, configured dimension and the dimension actually returned by a
live call. It exits non-zero if they disagree.

### 4. Start Milvus standalone

Milvus standalone is not a single container: it needs `etcd` for metadata and `MinIO` for object
storage.

```bash
docker compose -f deploy/compose/milvus.yml up -d
curl -f http://localhost:9091/healthz    # expect: OK
```

### 5. Migrate existing ChromaDB embeddings into Milvus

The migration reuses the embeddings already stored in Chroma, so no re-embedding cost is incurred.
This only works if the Chroma data was produced by the same embedding model that is configured
now - otherwise the dimensions will not match and you need to re-embed instead.

```bash
uv run python scripts/migrate_chroma_to_milvus.py --batch-size 1000
```

### 6. Verify parity between the two backends

```bash
uv run python scripts/benchmark_overlap.py \
  --queries evaluation/queries/smoke_queries.json \
  --top-k 5 \
  --out docs/benchmark.md
```

Acceptance criterion: **top-5 overlap between Chroma and Milvus >= 80%**.

### 7. Run the test suite

```bash
uv run ruff check .
uv run pytest -m "not integration"        # contract tests, no services required
uv run pytest -m integration              # requires a running Milvus and Ollama
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

The model layer follows the same shape, which is what makes the OpenAI/Ollama switch a one-line
change in `.env`:

```
api/embeddings/
├── base.py             # Embedder (ABC) + HashEmbedder for offline tests
├── openai_embedder.py  # hosted
└── ollama_embedder.py  # local
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
