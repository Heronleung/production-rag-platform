from __future__ import annotations

import pytest

from api.embeddings import HashEmbedder
from api.vectorstore.base import Chunk

EMBEDDING_DIM = 64


@pytest.fixture(scope="session")
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=EMBEDDING_DIM)


@pytest.fixture
def sample_chunks(embedder: HashEmbedder) -> list[Chunk]:
    texts = [
        ("Milvus supports HNSW and IVF_FLAT indexes.", "milvus.md", 0),
        ("A collection must be loaded before it can be searched.", "milvus.md", 1),
        ("Kubernetes horizontal pod autoscaling reacts to custom metrics.", "k8s.md", 0),
        ("RAGAS measures faithfulness and answer relevancy.", "ragas.md", 0),
    ]
    vectors = embedder.embed([text for text, _, _ in texts])
    return [
        Chunk(text=text, source=source, chunk_index=index, vector=vector)
        for (text, source, index), vector in zip(texts, vectors, strict=True)
    ]
