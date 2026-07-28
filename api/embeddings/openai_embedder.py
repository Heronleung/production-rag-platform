"""OpenAI embedding provider.

Requires ``OPENAI_API_KEY``. Install the optional dependency with
``uv sync --extra openai``.
"""

from __future__ import annotations

from api.config import settings
from api.embeddings.base import Embedder

MAX_BATCH = 128


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "The 'openai' package is not installed. Run `uv sync --extra openai`, "
                "or set EMBEDDING_PROVIDER=ollama to use local models instead."
            ) from exc

        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Set it in .env, or set EMBEDDING_PROVIDER=ollama "
                "to run entirely on local models."
            )

        self.model = model or settings.openai_embedding_model
        self.dim = dim or settings.embedding_dim
        url = base_url or settings.openai_base_url
        self._client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start : start + MAX_BATCH]
            response = self._client.embeddings.create(model=self.model, input=batch)
            # The API may return items out of order; sort defensively.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors
