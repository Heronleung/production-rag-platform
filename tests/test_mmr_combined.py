"""Regression tests for composing multi-query candidate scores with MMR."""

from api.retrieval.mmr import mmr_select
from api.vectorstore.base import SearchHit


def test_mmr_relevance_uses_original_query_not_variant_score() -> None:
    """A variant-favoured hit must not outrank the original-query match.

    Multi-query merges candidates by their best score against any generated
    query. The first candidate below therefore has the larger incoming score,
    but its vector is orthogonal to the original query. Standard MMR relevance
    must be recomputed from the original query vector.
    """
    candidates = [
        SearchHit(
            text="best match for a generated variant",
            source="doc.md",
            chunk_index=1,
            score=0.99,
            vector=[0.0, 1.0],
        ),
        SearchHit(
            text="best match for the original question",
            source="doc.md",
            chunk_index=0,
            score=0.70,
            vector=[1.0, 0.0],
        ),
    ]

    selected = mmr_select(
        query_vector=[1.0, 0.0],
        candidates=candidates,
        top_k=2,
        lambda_=0.5,
    )

    assert selected[0].chunk_index == 0


def test_lambda_one_sorts_by_original_query_relevance() -> None:
    candidates = [
        SearchHit("variant", "doc.md", 1, 0.99, [0.0, 1.0]),
        SearchHit("original", "doc.md", 0, 0.70, [1.0, 0.0]),
    ]

    selected = mmr_select([1.0, 0.0], candidates, top_k=2, lambda_=1.0)

    assert [hit.chunk_index for hit in selected] == [0, 1]
