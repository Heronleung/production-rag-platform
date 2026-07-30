"""``POST /query`` - retrieve context, then answer with the configured LLM.

Two response shapes behind one endpoint:

* ``stream: false`` -> a single :class:`QueryResponse` JSON body. Easiest to test
  and to call from scripts.
* ``stream: true`` (default) -> Server-Sent Events. The client sees citations
  first, then answer tokens as they are produced, then a terminating event.
  SSE is used instead of WebSockets because the traffic is one-way and it needs
  no protocol upgrade, so plain HTTP proxies and Kubernetes ingresses work.

SSE event contract::

    event: citations   data: {"citations": [...], "llm_model": "..."}
    event: token       data: {"text": "partial answer"}
    event: done        data: {"elapsed_seconds": 1.23}
    event: error       data: {"detail": "..."}

Errors raised mid-stream cannot change the HTTP status code, because the headers
have already been sent. They are therefore delivered as an ``error`` event and
the client must handle it explicitly.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_chat_model, get_embedder_singleton, get_vector_store
from api.embeddings import Embedder
from api.llm import ChatModel, Message
from api.logging_config import request_id_var
from api.retrieval.pipeline import retrieve as retrieval_pipeline
from api.schemas import Citation, QueryRequest, QueryResponse
from api.vectorstore.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer using only the numbered context "
    "passages provided. Cite the passages you rely on as [1], [2] and so on. If the "
    "context does not contain the answer, say so plainly instead of guessing."
)


def build_messages(question: str, hits: list[SearchHit]) -> list[Message]:
    context = "\n\n".join(
        f"[{index}] (source: {hit.source}#{hit.chunk_index})\n{hit.text}"
        for index, hit in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def _build_citations(hits: list[SearchHit]) -> list[Citation]:
    return [
        Citation(
            source=hit.source,
            chunk_index=hit.chunk_index,
            score=round(hit.score, 6),
            text=hit.text,
        )
        for hit in hits
    ]


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question over the ingested corpus",
    responses={
        200: {
            "content": {
                "text/event-stream": {},
                "application/json": {},
            },
            "description": "SSE stream when `stream` is true, otherwise a JSON body.",
        }
    },
)
def query(
    payload: QueryRequest,
    embedder: Embedder = Depends(get_embedder_singleton),
    store: VectorStore = Depends(get_vector_store),
    llm: ChatModel = Depends(get_chat_model),
):
    started = time.perf_counter()

    hits = retrieval_pipeline(
        query=payload.query,
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=payload.top_k,
        source_filter=payload.source_filter,
        use_mmr=payload.use_mmr,
        mmr_lambda=payload.mmr_lambda,
        mmr_fetch_k=payload.mmr_fetch_k,
        multi_query=payload.multi_query,
        multi_query_count=payload.multi_query_count,
    )

    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No context found. Ingest documents first, or relax source_filter.",
        )

    citations = _build_citations(hits)
    messages = build_messages(payload.query, hits)

    if not payload.stream:
        answer = llm.complete(messages, temperature=payload.temperature)
        elapsed = time.perf_counter() - started
        logger.info(
            "query completed",
            extra={
                "top_k": payload.top_k,
                "hits": len(hits),
                "streamed": False,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        return QueryResponse(
            query=payload.query,
            answer=answer,
            citations=citations,
            llm_model=llm.describe(),
            elapsed_seconds=round(elapsed, 3),
        )

    # The generator body runs after the response headers are sent. Starlette
    # iterates a sync generator through anyio's thread pool and runs every
    # next() call in a *fresh copy* of the context, so the ContextVar cannot be
    # used here: a token set in the first step cannot be reset in the last one
    # ("Token ... was created in a different Context"), and a value set inside
    # would not survive to the next step anyway. The id is therefore read once,
    # here, and passed explicitly to each log call below - JsonFormatter applies
    # `extra` over the ContextVar value, so the output is unchanged.
    request_id = request_id_var.get()

    def event_stream() -> Iterator[str]:
        try:
            yield _sse(
                "citations",
                {
                    "citations": [citation.model_dump() for citation in citations],
                    "llm_model": llm.describe(),
                },
            )
            for fragment in llm.stream(messages, temperature=payload.temperature):
                yield _sse("token", {"text": fragment})
            elapsed = time.perf_counter() - started
            logger.info(
                "query completed",
                extra={
                    "top_k": payload.top_k,
                    "hits": len(hits),
                    "streamed": True,
                    "elapsed_seconds": round(elapsed, 3),
                    "request_id": request_id,
                },
            )
            yield _sse("done", {"elapsed_seconds": round(elapsed, 3)})
        except Exception as exc:  # noqa: BLE001 - must surface inside the stream
            logger.exception("query stream failed", extra={"request_id": request_id})
            yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx-style proxies not to buffer, which would otherwise
            # hold tokens back and destroy the point of streaming.
            "X-Accel-Buffering": "no",
        },
    )
