"""Unified retrieval pipeline.

All of :mod:`api.routers.query` calls :func:`retrieve`; it never touches
the vector store or the retrieval strategies directly.  This keeps the router
thin and makes retrieval behaviour testable without HTTP.

Dispatch logic
--------------
* Both flags off (default): plain ``store.search()``.
* ``use_mmr=True``:         over-fetch with ``return_vectors=True``, then apply MMR.
* ``multi_query=True``:     expand question, search with each variant, merge.
* Both on:                  multi-query first (larger fetch), then MMR on the merged set.
"""

from __future__ import annotations

import logging

from api.embeddings import Embedder
from api.llm import ChatModel
from api.retrieval.mmr import mmr_select
from api.retrieval.multi_query import multi_query_search
from api.vectorstore.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    llm: ChatModel,
    top_k: int = 5,
    source_filter: str | None = None,
    use_mmr: bool = False,
    mmr_lambda: float = 0.5,
    mmr_fetch_k: int | None = None,
    multi_query: bool = False,
    multi_query_count: int = 3,
) -> list[SearchHit]:
    """Run the retrieval pipeline and return up to ``top_k`` hits.

    Parameters
    ----------
    query:
        The user’s question as a plain string.
    embedder:
        Embedding model; used for the main query (and for each variant when
        ``multi_query=True``).
    store:
        Vector store backend.
    llm:
        Chat model; used only for multi-query expansion, not for answering.
    top_k:
        Number of hits to return.
    source_filter:
        Restrict results to chunks from this source (exact match).
    use_mmr:
        If ``True``, over-fetch candidates and re-rank by MMR to reduce redundancy.
    mmr_lambda:
        Relevance/diversity trade-off for MMR.  ``1.0`` = pure relevance.
    mmr_fetch_k:
        Number of candidates to fetch before MMR selection.  Defaults to
        ``top_k * 4``.
    multi_query:
        If ``True``, generate query variants and merge results before returning.
    multi_query_count:
        Number of LLM-generated query variants (in addition to the original).
    """
    # The query vector is needed in all paths (vanilla, MMR, and as the
    # reference vector for MMR when multi_query is also on).
    query_vector = embedder.embed_query(query)
    fetch_k = mmr_fetch_k or (top_k * 4)

    if multi_query:
        hits = multi_query_search(
            question=query,
            llm=llm,
            embedder=embedder,
            store=store,
            # Over-fetch so MMR has enough candidates if both flags are on.
            top_k=fetch_k if use_mmr else top_k,
            source_filter=source_filter,
            count=multi_query_count,
            return_vectors=use_mmr,
        )
    else:
        hits = store.search(
            query_vector,
            top_k=fetch_k if use_mmr else top_k,
            source_filter=source_filter,
            return_vectors=use_mmr,
        )

    if use_mmr:
        hits = mmr_select(query_vector, hits, top_k=top_k, lambda_=mmr_lambda)

    return hits[:top_k]
