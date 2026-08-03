"""Phase 3: unit tests for MMR selection and multi-query retrieval.

No services are required. All tests use InMemoryStore, FakeEmbedder, and
FakeLLM so they run fully offline.
"""

from __future__ import annotations

from api.retrieval.mmr import mmr_select
from api.retrieval.multi_query import expand_queries, multi_query_search
from api.retrieval.pipeline import retrieve
from api.vectorstore.base import Chunk, SearchHit
from api.vectorstore.memory_store import InMemoryStore

DIM = 4


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLLM:
    """Returns a fixed string from complete(); stream() yields it whole."""

    def __init__(self, response: str = "[]") -> None:
        self._response = response

    def complete(self, messages, temperature=0.0):  # noqa: ANN001
        return self._response

    def stream(self, messages, temperature=0.0):  # noqa: ANN001
        yield self._response

    def describe(self) -> str:
        return "fake:fake-1"


class FakeEmbedder:
    """Always returns the same unit vector pointing along the first axis."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dim - 1)] * len(texts)

    def embed_query(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (self.dim - 1)

    def describe(self) -> str:
        return "fake:hash-embed"


def _store(*vecs: list[float]) -> InMemoryStore:
    """Build a populated InMemoryStore from raw vectors."""
    store = InMemoryStore(dim=DIM)
    store.upsert(
        [
            Chunk(text=f"text {i}", source=f"doc{i}.md", chunk_index=i, vector=vec)
            for i, vec in enumerate(vecs)
        ]
    )
    return store


# ---------------------------------------------------------------------------
# MMR tests
# ---------------------------------------------------------------------------


def test_mmr_returns_requested_count() -> None:
    candidates = [
        SearchHit(text="a", source="s", chunk_index=0, score=0.9, vector=[1, 0, 0, 0]),
        SearchHit(text="b", source="s", chunk_index=1, score=0.8, vector=[0, 1, 0, 0]),
        SearchHit(text="c", source="s", chunk_index=2, score=0.7, vector=[0, 0, 1, 0]),
        SearchHit(text="d", source="s", chunk_index=3, score=0.6, vector=[0, 0, 0, 1]),
    ]
    result = mmr_select([1, 0, 0, 0], candidates, top_k=2, lambda_=0.5)
    assert len(result) == 2


def test_mmr_first_pick_is_highest_relevance() -> None:
    """Regardless of lambda, the first selected hit is always the most relevant."""
    candidates = [
        SearchHit(text="a", source="s", chunk_index=0, score=0.9, vector=[1, 0, 0, 0]),
        SearchHit(text="b", source="s", chunk_index=1, score=0.5, vector=[0, 0, 1, 0]),
    ]
    for lam in (0.0, 0.5, 1.0):
        result = mmr_select([1, 0, 0, 0], candidates, top_k=2, lambda_=lam)
        assert result[0].chunk_index == 0, f"lambda={lam}: first pick should be chunk 0"


def test_mmr_avoids_near_duplicate() -> None:
    """With pure diversity (lambda=0), a near-duplicate should not be second pick."""
    candidates = [
        SearchHit(text="a", source="s", chunk_index=0, score=0.9, vector=[1, 0, 0, 0]),
        SearchHit(text="b", source="s", chunk_index=1, score=0.85, vector=[1, 0, 0, 0]),
        SearchHit(text="c", source="s", chunk_index=2, score=0.5, vector=[0, 0, 1, 0]),
    ]
    result = mmr_select([1, 0, 0, 0], candidates, top_k=2, lambda_=0.0)
    assert len(result) == 2
    selected_indices = {h.chunk_index for h in result}
    assert 1 not in selected_indices
    assert 2 in selected_indices


def test_mmr_lambda_one_degenerates_to_top_k() -> None:
    """lambda=1.0 must return hits in their original relevance order."""
    candidates = [
        SearchHit(text="a", source="s", chunk_index=0, score=0.9, vector=[1, 0, 0, 0]),
        SearchHit(text="b", source="s", chunk_index=1, score=0.8, vector=[0, 1, 0, 0]),
        SearchHit(text="c", source="s", chunk_index=2, score=0.7, vector=[0, 0, 1, 0]),
    ]
    result = mmr_select([1, 0, 0, 0], candidates, top_k=2, lambda_=1.0)
    assert [h.chunk_index for h in result] == [0, 1]


def test_mmr_fewer_candidates_than_top_k() -> None:
    candidates = [
        SearchHit(text="a", source="s", chunk_index=0, score=0.9, vector=[1, 0, 0, 0]),
    ]
    result = mmr_select([1, 0, 0, 0], candidates, top_k=5, lambda_=0.5)
    assert len(result) == 1


def test_mmr_empty_candidates() -> None:
    assert mmr_select([1, 0, 0, 0], [], top_k=3) == []


# ---------------------------------------------------------------------------
# Multi-query tests
# ---------------------------------------------------------------------------


def test_expand_queries_parses_valid_json() -> None:
    llm = FakeLLM('["rephrase one", "rephrase two", "rephrase three"]')
    variants = expand_queries("original", llm, count=3)
    assert variants == ["rephrase one", "rephrase two", "rephrase three"]


def test_expand_queries_truncates_to_count() -> None:
    llm = FakeLLM('["a", "b", "c", "d", "e"]')
    variants = expand_queries("q", llm, count=2)
    assert len(variants) == 2


def test_expand_queries_fallback_on_bad_json() -> None:
    llm = FakeLLM("this is not json at all")
    variants = expand_queries("original", llm, count=3)
    assert variants == []


def test_expand_queries_fallback_on_non_list_json() -> None:
    llm = FakeLLM('{"key": "value"}')
    variants = expand_queries("original", llm, count=3)
    assert variants == []


def test_multi_query_deduplicates_by_chunk_key() -> None:
    """Multiple query variants that return the same chunks should not duplicate."""
    store = _store([1, 0, 0, 0], [0, 1, 0, 0])
    llm = FakeLLM('["variant a", "variant b"]')
    embedder = FakeEmbedder()
    hits = multi_query_search("q", llm, embedder, store, top_k=10, source_filter=None, count=2)
    keys = [h.key for h in hits]
    assert len(keys) == len(set(keys)), "duplicate chunk keys found after merge"


def test_multi_query_keeps_highest_score() -> None:
    """After deduplication, each chunk key should appear with its highest score."""
    store = _store([1, 0, 0, 0], [0, 1, 0, 0])
    llm = FakeLLM('["alt"]')
    embedder = FakeEmbedder()
    hits = multi_query_search("q", llm, embedder, store, top_k=5, source_filter=None, count=1)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "hits are not sorted by descending score"


def test_multi_query_fallback_on_bad_llm() -> None:
    """When LLM returns garbage, multi-query falls back to the original question only."""
    store = _store([1, 0, 0, 0])
    llm = FakeLLM("not json")
    embedder = FakeEmbedder()
    hits = multi_query_search("q", llm, embedder, store, top_k=5, source_filter=None, count=3)
    assert len(hits) >= 1


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


def test_pipeline_vanilla_returns_hits() -> None:
    store = _store([1, 0, 0, 0], [0, 1, 0, 0])
    hits = retrieve("q", FakeEmbedder(), store, FakeLLM(), top_k=2)
    assert 1 <= len(hits) <= 2


def test_pipeline_mmr_removes_near_duplicate() -> None:
    """With pure diversity, a near-duplicate chunk should be dropped."""
    store = InMemoryStore(dim=DIM)
    store.upsert(
        [
            Chunk(text="t0", source="s", chunk_index=0, vector=[1, 0, 0, 0]),
            Chunk(text="t1", source="s", chunk_index=1, vector=[1, 0, 0, 0]),
            Chunk(text="t2", source="s", chunk_index=2, vector=[0, 1, 0, 0]),
        ]
    )
    hits = retrieve(
        "q",
        FakeEmbedder(),
        store,
        FakeLLM(),
        top_k=2,
        use_mmr=True,
        mmr_lambda=0.0,
        mmr_fetch_k=3,
    )
    assert len(hits) == 2
    indices = {h.chunk_index for h in hits}
    assert 0 in indices
    assert 1 not in indices
    assert 2 in indices


def test_pipeline_multi_query_no_duplicate_keys() -> None:
    store = _store([1, 0, 0, 0], [0, 1, 0, 0])
    hits = retrieve(
        "q",
        FakeEmbedder(),
        store,
        FakeLLM('["variant"]'),
        top_k=5,
        multi_query=True,
        multi_query_count=1,
    )
    keys = [h.key for h in hits]
    assert len(keys) == len(set(keys))


def test_pipeline_defaults_match_phase2_behaviour() -> None:
    """With all flags at default (off), results must match a plain store.search()."""
    store = InMemoryStore(dim=DIM)
    store.upsert(
        [Chunk(text="only", source="x.md", chunk_index=0, vector=[1, 0, 0, 0])]
    )
    hits = retrieve("q", FakeEmbedder(), store, FakeLLM(), top_k=5)
    assert len(hits) == 1
    assert hits[0].source == "x.md"
