# Production RAG Platform

A production-oriented Retrieval-Augmented Generation platform built around **Milvus**, **FastAPI**, **Next.js**, and local **Ollama** models. OpenAI remains available behind the same provider interfaces.

The repository is developed in independently verified phases. Work through Phase 6 is complete: ingestion, retrieval, evaluation, the frontend, container images, and a local Kubernetes deployment are implemented and tested. CI/CD and observability remain future phases.

## Roadmap status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Project skeleton and tooling | complete |
| 1 | Milvus behind a `VectorStore` interface | complete |
| 2 | FastAPI ingestion/query API and SSE streaming | complete |
| 3 | MMR and multi-query retrieval | complete |
| 4 | Deterministic evaluation and regression gate | complete |
| 5 | Next.js frontend | complete |
| 6 | Containers, kind, Helm, ingress, security and rollback | complete |
| 7 | CI/CD pipeline | not started |
| 8 | Observability and final operational documentation | not started |

See [`docs/roadmap.md`](docs/roadmap.md) for the broader plan.

## Architecture

```text
Browser
  │
  ▼
Next.js web application
  │  /api/ready, /api/ingest, /api/query
  ▼
FastAPI
  ├── ingestion/chunking.py
  ├── Ollama or OpenAI embeddings
  ├── Milvus vector store
  ├── vanilla / MMR / multi-query retrieval
  └── Ollama or OpenAI chat generation
```

The core boundaries are interfaces:

- `api.embeddings.Embedder`
- `api.llm.ChatModel`
- `api.vectorstore.base.VectorStore`

Tests replace external providers with deterministic in-memory implementations, while integration tests exercise Milvus separately.

## Requirements

- Python 3.11+
- `uv`
- Node.js 22+
- Docker
- Ollama
- For Kubernetes: kind, kubectl, and Helm 3

## Local API quick start

Install the locked development environment:

```bash
uv sync --frozen --extra dev
cp .env.example .env
```

Start Ollama and pull the default local models:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

If port 11434 is already in use, Ollama is already running; do not start a second instance.

Start Milvus:

```bash
docker compose -f deploy/compose/milvus.yml up -d
curl -f http://localhost:9091/healthz
```

Run the API with a scoped reloader:

```bash
uv run python -m uvicorn api.main:app \
  --reload \
  --reload-dir api \
  --reload-dir ingestion
```

The scoped reload directories avoid traversing root-owned Docker volumes under `deploy/compose/volumes/`.

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Process-only liveness; never touches dependencies |
| GET | `/readyz` | Checks the embedder, configured chat model, and Milvus; returns 503 when any is unavailable |
| POST | `/ingest` | Upload and index a PDF, Markdown, or text file |
| POST | `/query` | Retrieve context and answer; SSE by default |

```bash
curl -sS http://localhost:8000/readyz | python3 -m json.tool

curl -F 'file=@docs/roadmap.md' \
  http://localhost:8000/ingest

curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which indexes does Milvus support?","top_k":5}'
```

A successful SSE response emits `citations`, one or more `token` events, and `done`. Mid-stream failures are emitted as `error` events because the HTTP headers have already been sent.

Every request receives an `X-Request-ID`. An inbound value is preserved; otherwise one is generated and attached to response headers, structured logs, and error bodies.

## Retrieval and evaluation

Retrieval flags are disabled by default, preserving the Phase 2 behavior:

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"Milvus persistence",
    "top_k":5,
    "use_mmr":true,
    "multi_query":true
  }'
```

Run the deterministic evaluation and regression gate:

```bash
uv run python scripts/evaluate.py
```

The golden dataset compares vanilla, MMR, multi-query, and combined retrieval. Optional Ragas integration remains deferred rather than being required for the deterministic gate.

## Frontend

```bash
cd web
npm ci
npm run dev
```

Open <http://localhost:3000>. The UI supports readiness checks, document ingestion, SSE token streaming, citations, Stop, and Retry.

Production verification:

```bash
npm run lint
npm run typecheck
npm test
npm run build
npm audit
```

## Containers and Kubernetes

Full deployment and recovery instructions are in [`docs/deployment.md`](docs/deployment.md).

### Local Compose application layer

```bash
docker compose -f deploy/compose/app.yml up --build -d
curl -f http://localhost:8000/healthz
curl -f http://localhost:3000/healthz
curl -f http://localhost:3000/api/ready
```

### kind and Helm

The verified kind profile uses the host-accessible Ollama model `qwen2.5:3b`:

```bash
ollama pull qwen2.5:3b

bash deploy/scripts/kind-up.sh
bash deploy/scripts/build-and-load.sh
```

Add the local hostname once:

```text
127.0.0.1 rag.local
```

Then open <http://rag.local:8080>.

The verified readiness response contains:

- `embedder: ollama:nomic-embed-text`
- `chat_model: ollama:qwen2.5:3b`
- `vector_store: rag_chunks`

Ollama readiness calls `/api/tags` and confirms that the configured chat model exists. A missing model returns 503, keeps the replacement pod out of Service endpoints, and prevents a bad Helm rollout.

### Immutable deployment

```bash
TAG="phase6-$(git rev-parse --short=12 HEAD)"

docker build -f Dockerfile.api -t "rag-api:${TAG}" .
docker build -f web/Dockerfile -t "rag-web:${TAG}" web

kind load docker-image --name rag \
  "rag-api:${TAG}" \
  "rag-web:${TAG}"

helm upgrade --install rag deploy/helm/rag-platform \
  --namespace rag \
  --create-namespace \
  -f deploy/helm/rag-platform/values-kind.yaml \
  --set-string "api.image.tag=${TAG}" \
  --set-string "web.image.tag=${TAG}" \
  --wait \
  --timeout 5m
```

Smoke check:

```bash
bash deploy/scripts/smoke-k8s.sh
```

### Security defaults

The API and web containers:

- Run as non-root users
- Use read-only root filesystems
- Drop all Linux capabilities
- Deny privilege escalation
- Use the `RuntimeDefault` seccomp profile
- Mount writable temporary directories only where required

Provider secrets are referenced through an existing Kubernetes Secret and are not stored in chart values or image history.

NetworkPolicy remains disabled for the local kind profile because host dependency CIDRs vary by Docker installation. The chart includes the policy template; enable it only after configuring stable `networkPolicy.allowedExternalCIDRs` for Milvus and Ollama.

## Verification

Python:

```bash
uv run ruff check api tests
uv run pytest -m "not integration"
uv run pytest -m integration
```

Helm:

```bash
helm lint deploy/helm/rag-platform \
  -f deploy/helm/rag-platform/values-kind.yaml

helm template rag deploy/helm/rag-platform \
  --namespace rag \
  -f deploy/helm/rag-platform/values-kind.yaml \
  > /tmp/rag-rendered.yaml

kubectl apply --dry-run=server \
  --namespace rag \
  -f /tmp/rag-rendered.yaml
```

Warnings about `kubectl.kubernetes.io/last-applied-configuration` are expected during the server-side dry run because the live resources are managed by Helm rather than `kubectl apply`.

## Documentation

- [`docs/api.md`](docs/api.md) — HTTP API and SSE contract
- [`docs/retrieval.md`](docs/retrieval.md) — MMR and multi-query retrieval
- [`docs/deployment.md`](docs/deployment.md) — containers, kind, Helm, ingress, security, recovery and rollback
- [`docs/roadmap.md`](docs/roadmap.md) — phase roadmap
