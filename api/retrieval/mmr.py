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

    Parameters
    ----------
    query_vector:
        Embedding of the user’s question.
    candidates:
        Hits from an initial over-fetch.  Each hit **must** have a non-empty
        ``vector`` field (request from the store with ``return_vectors=True``).
    top_k:
        Number of hits to select.
    lambda_:
        Trade-off coefficient.  ``1.0`` returns the top-relevance ranking;
        ``0.0`` maximises diversity.  Defaults to ``0.5``.

    Returns
    -------
    list[SearchHit]
        Hits in MMR-selected order (first hit is always the most relevant).
    """
    if not candidates:
        return []
    # Pure relevance: skip the expensive per-step max computation.
    if lambda_ >= 1.0:
        return candidates[:top_k]

    selected: list[SearchHit] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick is always the highest-relevance candidate so the
            # answer remains grounded even at low lambda values.
            best = remaining[0]
        else:
            selected_vecs = [h.vector for h in selected]
            best = max(
                remaining,
                key=lambda h: (
                    lambda_ * h.score
                    - (1.0 - lambda_) * max(_cosine(h.vector, sv) for sv in selected_vecs)
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return selected
