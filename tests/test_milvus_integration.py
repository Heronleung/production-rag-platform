"""Integration tests against a real Milvus instance.

Run them with:
    docker compose -f deploy/compose/milvus.yml up -d
    uv run pytest -m integration

They are excluded from the default run because CI should not need a database to
validate the interface contract.
"""

from __future__ import annotations

import os
import uuid

import pytest

from api.embeddings import HashEmbedder
from api.vectorstore.base import Chunk

from .conftest import EMBEDDING_DIM

pytestmark = pytest.mark.integration


@pytest.fixture
def milvus_store():
    pytest.importorskip("pymilvus")
    from api.vectorstore.milvus_store import MilvusStore

    store = MilvusStore(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        collection=f"test_{uuid.uuid4().hex[:8]}",
        dim=EMBEDDING_DIM,
    )
    try:
        yield store
    finally:
        store.drop()


def test_insert_then_search(milvus_store, sample_chunks: list[Chunk],
                            embedder: HashEmbedder) -> None:
    milvus_store.upsert(sample_chunks)
    milvus_store.wait_for_index()
    milvus_store.client.load_collection(milvus_store.collection)

    assert milvus_store.count() == len(sample_chunks)

    query = embedder.embed(["Which indexes does Milvus support?"])[0]
    hits = milvus_store.search(query, top_k=2)
    assert hits
    assert hits[0].source == "milvus.md"


def test_source_filter(milvus_store, sample_chunks: list[Chunk],
                       embedder: HashEmbedder) -> None:
    milvus_store.upsert(sample_chunks)
    milvus_store.wait_for_index()
    milvus_store.client.load_collection(milvus_store.collection)

    query = embedder.embed(["Milvus"])[0]
    hits = milvus_store.search(query, top_k=5, source_filter="k8s.md")
    assert hits
    assert {hit.source for hit in hits} == {"k8s.md"}
