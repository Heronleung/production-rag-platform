# HTTP API (Phase 2)

FastAPI application: `api/main.py`. Interactive docs at `http://localhost:8000/docs`.

```bash
uv sync --extra dev
uv run uvicorn api.main:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness. Never touches a dependency, so a slow Milvus cannot trigger a restart loop. |
| GET | `/readyz` | Readiness. Checks the embedder and the vector store; returns **503** when either is unreachable. |
| POST | `/ingest` | Upload a `.pdf` / `.md` / `.txt` file; it is chunked, embedded and written to the vector store. |
| POST | `/query` | Retrieve context and generate an answer. SSE stream by default. |
| GET | `/openapi.json`, `/docs` | Generated schema and Swagger UI. |

Every response carries an `X-Request-ID` header. An inbound `X-Request-ID` is
preserved so a gateway's correlation id survives; otherwise one is generated.
The same id appears in every JSON log line produced while handling the request,
and in the body of every error response.

## POST /ingest

```bash
curl -F 'file=@docs/roadmap.md' 'http://localhost:8000/ingest?chunk_size=1000&chunk_overlap=50'
```

```json
{
  "source": "roadmap.md",
  "chunks_written": 12,
  "embedding_model": "ollama:nomic-embed-text (dim=768)",
  "elapsed_seconds": 1.84
}
```

Status codes: `201` written, `400` empty file or bad chunk parameters, `413`
larger than 25MB, `415` unsupported extension, `422` no extractable text (a
scanned PDF with no text layer lands here).

Chunking is `ingestion.chunking.split_text`, the same function
`scripts/build_chroma_baseline.py` uses, so an HTTP-ingested document is
indistinguishable from an offline-ingested one.

## POST /query

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Which indexes does Milvus support?", "top_k": 5}'
```

SSE event sequence:

```
event: citations   data: {"citations": [...], "llm_model": "ollama:qwen2.5:1.5b"}
event: token       data: {"text": "Milvus "}
event: token       data: {"text": "supports "}
event: done        data: {"elapsed_seconds": 4.12}
```

Citations are sent **before** the first token, so the UI can render sources
immediately while generation is still running.

Pass `"stream": false` for a single JSON body instead:

```json
{
  "query": "Which indexes does Milvus support?",
  "answer": "Milvus supports HNSW and IVF_FLAT [1].",
  "citations": [{ "source": "milvus.md", "chunk_index": 0, "score": 0.83, "text": "..." }],
  "llm_model": "ollama:qwen2.5:1.5b",
  "elapsed_seconds": 3.9
}
```

`404` is returned when retrieval finds nothing at all - an empty corpus or an
over-narrow `source_filter` - instead of letting the model answer from thin air.

## Design decisions

**SSE rather than WebSockets.** Traffic is one-way (server to client), so SSE
needs no protocol upgrade and works through ordinary HTTP proxies and
Kubernetes ingresses. `X-Accel-Buffering: no` is set so nginx does not buffer
tokens and defeat the point.

**Errors inside a stream are events, not status codes.** Once the first byte is
sent the status code is fixed at 200, so a mid-stream failure arrives as
`event: error` and the client must handle it.

**Blocking handlers are `def`, not `async def`.** The embedder, the vector store
and the LLM are synchronous. Declared as `def`, FastAPI runs them in a worker
thread and the event loop stays free; declared `async def` they would block
every other connection.

**Dependencies are cached singletons.** `api/dependencies.py` builds the
embedder, store and chat model once (`lru_cache`) because the Milvus client
opens a connection and loads the collection into memory. They are injected with
`Depends`, which is also the seam the tests replace with `HashEmbedder` +
`InMemoryStore` + a fake chat model - `tests/test_api.py` runs the full HTTP
path with no services running.

**Startup never fails on a dependency.** Clients are built lazily on first use,
so the API still starts when Milvus or Ollama is down and reports the problem
through `/readyz` instead of crash-looping.
