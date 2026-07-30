"""Multi-query retrieval expansion.

The core idea: the same semantic content can be described in many ways, and a
single embedding may miss chunks that are relevant but phrased differently.
Sending several reformulated versions of the question and merging the results
improves recall without changing the vector store or the embedding model.

Pipeline
--------
1. Ask the LLM to generate ``count`` alternative phrasings (JSON array).
2. Embed the original question and all variants.
3. Search the vector store with each embedding.
4. Merge all hit sets, deduplicating by ``chunk.key`` and keeping the highest
   score seen for each chunk across all queries.
5. Return the merged set sorted by score, truncated to ``top_k``.

If the LLM returns unparseable output, the function logs a warning and falls
back to the original question only — the caller always gets a valid result.
"""

from __future__ import annotations

import json
import logging

from api.embeddings import Embedder
from api.llm import ChatModel
from api.vectorstore.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)

_EXPANSION_PROMPT = (
    "You are a query expansion assistant. Given a user question, generate {n} "
    "alternative phrasings that cover different angles or terminology. "
    "Return ONLY a valid JSON array of strings with no explanation. Example:\n"
    '["alternative 1", "alternative 2"]\n\n'
    "User question: {question}"
)


def expand_queries(question: str, llm: ChatModel, count: int = 3) -> list[str]:
    """Return up to ``count`` LLM-generated rephrases of ``question``.

    Returns an empty list when expansion fails (LLM error or unparseable JSON).
    The caller is responsible for always including the original question.
    """
    prompt = _EXPANSION_PROMPT.format(n=count, question=question)
    try:
        raw = llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.7,  # slight randomness for variety
        )
        variants: object = json.loads(raw)
        if not isinstance(variants, list):
            raise ValueError("LLM did not return a JSON array")
        cleaned = [str(v).strip() for v in variants if str(v).strip()]
        logger.debug("multi-query expanded to %d variants", len(cleaned))
        return cleaned[:count]
    except Exception:  # noqa: BLE001
        logger.warning("multi-query expansion failed, falling back to original query only")
        return []


def multi_query_search(
    question: str,
    llm: ChatModel,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
    source_filter: str | None,
    count: int = 3,
    return_vectors: bool = False,
) -> list[SearchHit]:
    """Search with the original question plus LLM-generated variants.

    Parameters
    ----------
    question:
        The user’s original question.
    llm:
        Chat model used only for query expansion, not for answering.
    embedder:
        Embedding model; called once per query variant.
    store:
        Vector store to search.
    top_k:
        Maximum number of hits to return after merging.
    source_filter:
        Forwarded verbatim to :meth:`VectorStore.search`.
    count:
        How many variants to generate.
    return_vectors:
        When ``True``, each returned hit will carry its stored vector.  Required
        when the caller intends to run MMR on the merged results.
    """
    variants = expand_queries(question, llm, count)
    queries = [question] + variants
    logger.info("multi-query: searching with %d queries total", len(queries))

    best: dict[str, SearchHit] = {}
    for q in queries:
        vec = embedder.embed_query(q)
        hits = store.search(
            vec,
            top_k=top_k,
            source_filter=source_filter,
            return_vectors=return_vectors,
        )
        for hit in hits:
            # Keep the full SearchHit (including vector) for the highest-scoring
            # occurrence of each chunk across all query variants.
            if hit.key not in best or hit.score > best[hit.key].score:
                best[hit.key] = hit

    merged = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return merged[:top_k]
