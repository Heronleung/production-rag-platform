"""Embedding providers.

Two real implementations live side by side in this package:

* :class:`~api.embeddings.openai_embedder.OpenAIEmbedder` - hosted API.
* :class:`~api.embeddings.ollama_embedder.OllamaEmbedder` - local models.

Both satisfy :class:`~api.embeddings.base.Embedder`, so callers use
:func:`get_embedder` and never import a concrete class. Switching provider is a
one-line change in ``.env`` and requires no code change anywhere else.
"""

from __future__ import annotations

from api.config import Provider, settings
from api.embeddings.base import Embedder, HashEmbedder

__all__ = [
    "Embedder",
    "HashEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
]


def __getattr__(name: str):
    """Import concrete providers lazily so that missing optional dependencies
    only fail when that provider is actually requested."""
    if name == "OpenAIEmbedder":
        from api.embeddings.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder
    if name == "OllamaEmbedder":
        from api.embeddings.ollama_embedder import OllamaEmbedder

        return OllamaEmbedder
    raise AttributeError(name)


def get_embedder(
    provider: Provider | None = None,
    model: str | None = None,
    dim: int | None = None,
) -> Embedder:
    """Build the embedder for the configured provider.

    Args:
        provider: ``"openai"`` or ``"ollama"``. Defaults to ``EMBEDDING_PROVIDER``.
        model: Overrides the provider's configured model name.
        dim: Overrides the resolved embedding dimension.
    """
    provider = provider or settings.embedding_provider

    if provider == "openai":
        from api.embeddings.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(model=model, dim=dim)

    if provider == "ollama":
        from api.embeddings.ollama_embedder import OllamaEmbedder

        return OllamaEmbedder(model=model, dim=dim)

    raise ValueError(f"Unknown embedding provider: {provider!r}. Use 'openai' or 'ollama'.")
