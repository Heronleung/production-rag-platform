"""Contract tests for the embedding providers.

The offline tests run everywhere. The provider tests are marked ``integration``
and skip themselves when the provider is not reachable, so a machine with no
Ollama daemon and no OpenAI key still gets a green default test run.
"""

from __future__ import annotations

import os

import pytest

from api.embeddings import HashEmbedder, get_embedder


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dim=32)
    first = embedder.embed(["Milvus supports HNSW"])
    second = embedder.embed(["Milvus supports HNSW"])
    assert first == second


def test_hash_embedder_reports_its_dimension() -> None:
    embedder = HashEmbedder(dim=32)
    assert embedder.probe_dim() == 32
    assert len(embedder.embed(["a", "b", "c"])) == 3


def test_embed_preserves_input_order() -> None:
    embedder = HashEmbedder(dim=32)
    texts = ["alpha", "beta", "gamma"]
    batched = embedder.embed(texts)
    individually = [embedder.embed_query(text) for text in texts]
    assert batched == individually


def test_empty_input_returns_empty_list() -> None:
    assert HashEmbedder(dim=32).embed([]) == []


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder(provider="cohere")  # type: ignore[arg-type]


@pytest.mark.integration
def test_ollama_embedder_matches_configured_dimension() -> None:
    from api.embeddings.ollama_embedder import OllamaEmbedder

    embedder = OllamaEmbedder()
    if not embedder.is_available():
        pytest.skip(f"Ollama is not running or '{embedder.model}' is not pulled.")
    assert embedder.probe_dim() == embedder.dim


@pytest.mark.integration
def test_openai_embedder_matches_configured_dimension() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")
    pytest.importorskip("openai")
    from api.embeddings.openai_embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder()
    assert embedder.probe_dim() == embedder.dim
