"""Maximal Marginal Relevance (MMR) selection.

Given a query vector and a set of candidate :class:`~api.vectorstore.base.SearchHit`
objects that carry their own stored vectors, greedily selects ``top_k`` hits
that balance relevance against redundancy.

Formula (Carbonell & Goldstein, 1998)::

    score(d) = λ · sim(d, query) − (1−λ) · max_{s ∈ S} sim(d, s)

where *S* is the set of already-selected hits and *sim* is cosine similarity.

* ``λ = 1.0`` — pure relevance; degenerates to plain top-k ranking.
* ``λ = 0.0`` — pure diversity; first pick is still the best match.
* ``λ = 0.5`` — default; balances answer completeness against redundancy.

The algorithm is O(candidates · selected) per step and typically runs on a
few dozen hits, so no vectorised implementation is needed.
"""

from __future__ import annotations

import math

from api.vectorstore.base import SearchHit


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select(
    query_vector: list[float],
    candidates: list[SearchHit],
    top_k: int,
    lambda_: float = 0.5,
) -> list[SearchHit]:
    """Return up to ``top_k`` hits selected by the MMR algorithm.

    Candidate ``score`` cannot be used as original-query relevance here. In a
    combined multi-query + MMR pipeline that score is the best similarity seen
    against *any generated query variant*. MMR's formula specifically requires
    similarity to ``query_vector``, so relevance is recomputed from each stored
    candidate vector. This is also why the first pick cannot simply trust the
    incoming candidate order.
    """
    if not candidates:
        return []

    relevance = {hit.key: _cosine(query_vector, hit.vector) for hit in candidates}

    # Pure relevance: return the ranking for the original query, not the
    # incoming order (which may be ordered by a multi-query variant score).
    if lambda_ >= 1.0:
        return sorted(candidates, key=lambda hit: relevance[hit.key], reverse=True)[:top_k]

    selected: list[SearchHit] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick is always the candidate most relevant to the original
            # question, even when the candidate pool came from multi-query.
            best = max(remaining, key=lambda hit: relevance[hit.key])
        else:
            selected_vecs = [hit.vector for hit in selected]
            best = max(
                remaining,
                key=lambda hit: (
                    lambda_ * relevance[hit.key]
                    - (1.0 - lambda_)
                    * max(_cosine(hit.vector, vector) for vector in selected_vecs)
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected
