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
| 2 | FastAPI backend (`/ingest`, `/query`, SSE streaming) | complete |
| 3 | Retrieval quality (MMR, multi-query) | in progress |
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

If `ollama serve` reports `address already in use` on port 11434, Ollama is already running as a
service - skip it and go straight to the pulls.

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

## Phase 2 quick start: the HTTP API

Full endpoint reference and design notes: **`docs/api.md`**.

```bash
uv sync --extra dev                       # pulls in fastapi, uvicorn, python-multipart
uv run uvicorn api.main:app --reload \
  --reload-dir api --reload-dir ingestion  # http://localhost:8000/docs
```

The `--reload-dir` flags are not optional in practice. Without them WatchFiles walks the entire
project, including `deploy/compose/volumes/`, where the etcd container creates root-owned
directories: traversal then fails with `Permission denied`, the reloader process dies and leaves
the server child orphaned. Watching only the source packages also means no pointless restarts on
docs or deploy changes.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness. Never touches a dependency. |
| GET | `/readyz` | Readiness. Checks embedder + vector store, returns 503 when either is down. |
| POST | `/ingest` | Upload a `.pdf`/`.md`/`.txt` file: chunk, embed, store. |
| POST | `/query` | Retrieve context and answer, streamed as SSE by default. |

```bash
# confirm the live dependencies before anything else
curl -s http://localhost:8000/readyz | python3 -m json.tool

# ingest a document
curl -F 'file=@docs/roadmap.md' http://localhost:8000/ingest

# ask a question (SSE stream; -N disables curl's buffering)
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Which indexes does Milvus support?", "top_k": 5}'

# same question, single JSON response instead of a stream
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Which indexes does Milvus support?", "stream": false}'
```

Every request gets an `X-Request-ID` (inbound header preserved, otherwise generated). It is
echoed on the response, attached to every structured JSON log line for that request, and
included in error bodies - so a user-reported failure can be traced end to end.

The API tests need no services at all: `api/dependencies.py` exposes the embedder, vector store
and chat model as injected singletons, and `tests/test_api.py` overrides them with
`HashEmbedder`, `InMemoryStore` and a fake chat model.

```bash
uv run pytest tests/test_api.py tests/test_chunking.py
```

---

## Phase 3: retrieval quality

Full design notes: **`docs/retrieval.md`**.

All Phase 3 flags are **off by default**. Existing clients need no changes.

```bash
# MMR: reduce redundancy among retrieved chunks
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Milvus index types", "top_k": 5, "use_mmr": true, "mmr_lambda": 0.6}'

# Multi-query: expand question into variants, merge results
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "approximate nearest neighbour", "top_k": 5, "multi_query": true}'

# Both combined: expand recall, then diversify
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Milvus persistence", "top_k": 5, "use_mmr": true, "multi_query": true}'
```

```bash
uv run pytest tests/test_retrieval.py tests/test_api.py tests/test_chunking.py
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

The retrieval layer (Phase 3) sits between the vector store and the router:

```
api/retrieval/
├── mmr.py          # Maximal Marginal Relevance selection
├── multi_query.py  # LLM-based query expansion and result merging
└── pipeline.py     # unified retrieve() called by the query router
```

The HTTP layer sits on top of both and adds nothing of its own beyond transport concerns:

```
api/
├── main.py             # app, request-id middleware, JSON error shape
├── dependencies.py     # cached embedder / store / LLM, the test seam
├── schemas.py          # Pydantic request+response models -> OpenAPI docs
├── logging_config.py   # structured JSON logs + request-id context
└── routers/
    ├── health.py       # /healthz, /readyz
    ├── ingest.py       # POST /ingest
    └── query.py        # POST /query (SSE); calls api.retrieval.pipeline.retrieve()
ingestion/
└── chunking.py         # splitter shared by POST /ingest and the Phase 1 scripts
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
