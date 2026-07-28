"""Embedding providers.

The rest of the codebase depends on the ``Embedder`` protocol only, so tests can
substitute a deterministic fake without any network access.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from api.config import settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Thin wrapper around the OpenAI embeddings endpoint."""

    def __init__(self, model: str | None = None, dim: int | None = None,
                 api_key: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class HashEmbedder:
    """Deterministic, dependency-free embedder for unit tests.

    It is not semantically meaningful. It only guarantees that identical text
    maps to an identical unit vector, which is enough to verify store contracts.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]
