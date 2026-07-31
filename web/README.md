# Phase 5 web frontend

Next.js App Router frontend for the Production RAG Platform. The browser talks only to same-origin `/api/*` route handlers; those handlers proxy to FastAPI using the server-only `RAG_API_URL`.

## Requirements

- Node.js 20.9 or newer
- The FastAPI backend and its Ollama/Milvus dependencies

## Run locally

```bash
# repository root, terminal 1
uv run python -m uvicorn api.main:app --reload \
  --reload-dir api --reload-dir ingestion

# terminal 2
cd web
cp .env.example .env.local
npm install
npm run dev
```

Open <http://localhost:3000>. `RAG_API_URL` defaults to `http://127.0.0.1:8000`, so copying the environment file is optional for the standard local setup.

## Verify

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

The pinned framework versions intentionally use Next.js 16.2.11 (Active LTS security release) and React 19.2.8. Commit the generated `package-lock.json` after the first networked `npm install`; do not hand-author a lockfile.

## Transport contract

| Browser route | FastAPI route | Notes |
| --- | --- | --- |
| `GET /api/health` | `GET /healthz` | Liveness passthrough |
| `GET /api/ready` | `GET /readyz` | Preserves 503 body with dependency details |
| `POST /api/ingest` | `POST /ingest` | Multipart form plus chunk query parameters |
| `POST /api/query` | `POST /query` | Streams the upstream SSE response body without buffering |

The query UI handles `citations`, `token`, `done`, and `error` events. It uses `AbortController` for Stop, preserves partial answers, and exposes backend request ids when available.

## Deliberate boundaries

- Vanilla retrieval remains the default; MMR and multi-query are opt-in controls.
- Citation text is rendered as text, never as HTML.
- Browser validation improves upload UX, but FastAPI remains the authority for file type, size, and chunk parameters.
- Container image, Kubernetes Service/Ingress, and production CORS tightening belong to Phase 6.
