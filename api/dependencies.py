"""Shared, lazily built singletons injected into the routers.

Why this module exists
----------------------
The embedder, vector store and chat model are expensive to construct: the Milvus
client opens a connection and loads the collection into memory, and the
embedder resolves its dimension. Building them per request would dominate
latency.

They are also the exact seam that the tests need to replace. Every router asks
for them through the FastAPI dependency system, so ``tests/test_api.py`` can
override them with a ``HashEmbedder`` and an ``InMemoryStore`` and exercise the
full HTTP path with no Milvus and no Ollama running.
"""

from __future__ import annotations

from functools import lru_cache

from api.config import settings
from api.embeddings import Embedder, get_embedder
from api.llm import ChatModel, get_llm
from api.vectorstore.base import VectorStore


@lru_cache(maxsize=1)
def get_embedder_singleton() -> Embedder:
    return get_embedder()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    # Imported here rather than at module scope so that importing the app does
    # not require pymilvus to be installed or Milvus to be reachable.
    from api.vectorstore.milvus_store import MilvusStore

    return MilvusStore(dim=settings.embedding_dim)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatModel:
    return get_llm()


def reset_singletons() -> None:
    """Drop the cached instances. Used by tests and by ``/readyz`` recovery."""
    get_embedder_singleton.cache_clear()
    get_vector_store.cache_clear()
    get_chat_model.cache_clear()
