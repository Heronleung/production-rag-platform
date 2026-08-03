"""Contract tests for the vector store abstraction.

The same assertions run against every backend. If Milvus passes this file, the
rest of the codebase can switch to it without any further change.
"""

from __future__ import annotations

import pytest
from conftest import EMBEDDING_DIM

from api.embeddings import HashEmbedder
from api.vectorstore.base import Chunk, VectorStore
from api.vectorstore.memory_store import InMemoryStore


@pytest.fixture
def store() -> VectorStore:
    return InMemoryStore(dim=EMBEDDING_DIM)


def test_upsert_returns_row_count(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    assert store.upsert(sample_chunks) == len(sample_chunks)
    assert store.count() == len(sample_chunks)


def test_upsert_is_idempotent(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    store.upsert(sample_chunks)
    store.upsert(sample_chunks)
    assert store.count() == len(sample_chunks), "re-writing the same chunks must not duplicate rows"


def test_search_returns_the_closest_chunk(
    store: VectorStore, sample_chunks: list[Chunk], embedder: HashEmbedder
) -> None:
    store.upsert(sample_chunks)
    query = embedder.embed(["Which indexes does Milvus support?"])[0]
    hits = store.search(query, top_k=2)

    assert hits, "search must return at least one hit"
    assert hits[0].source == "milvus.md"


def test_search_respects_top_k(
    store: VectorStore, sample_chunks: list[Chunk], embedder: HashEmbedder
) -> None:
    store.upsert(sample_chunks)
    query = embedder.embed(["anything"])[0]
    assert len(store.search(query, top_k=2)) <= 2


def test_scores_are_sorted_descending(
    store: VectorStore, sample_chunks: list[Chunk], embedder: HashEmbedder
) -> None:
    store.upsert(sample_chunks)
    query = embedder.embed(["Kubernetes autoscaling"])[0]
    scores = [hit.score for hit in store.search(query, top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_source_filter(
    store: VectorStore, sample_chunks: list[Chunk], embedder: HashEmbedder
) -> None:
    store.upsert(sample_chunks)
    query = embedder.embed(["Milvus"])[0]
    hits = store.search(query, top_k=5, source_filter="k8s.md")
    assert hits
    assert {hit.source for hit in hits} == {"k8s.md"}


def test_dimension_mismatch_is_rejected(store: VectorStore) -> None:
    bad = Chunk(text="wrong size", source="x.md", chunk_index=0, vector=[0.1, 0.2])
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert([bad])


def test_drop_clears_the_collection(store: VectorStore, sample_chunks: list[Chunk]) -> None:
    store.upsert(sample_chunks)
    store.drop()
    assert store.count() == 0
