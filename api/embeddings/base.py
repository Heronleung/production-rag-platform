"""The embedder abstraction shared by every provider."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors.

    ``name`` identifies the provider, ``model`` the concrete model, and ``dim``
    the vector width. All three are recorded in benchmarks and in the Milvus
    collection description so results stay reproducible.
    """

    name: str = "base"
    model: str = ""
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def probe_dim(self) -> int:
        """Ask the provider for a real vector and report its width.

        Use this to verify configuration rather than trusting the lookup table.
        """
        return len(self.embed_query("dimension probe"))

    def describe(self) -> str:
        return f"{self.name}:{self.model} (dim={self.dim})"


class HashEmbedder(Embedder):
    """Deterministic, offline embedder used by the unit tests.

    It carries no semantic meaning. It only guarantees that identical text maps
    to an identical unit vector and that similar token sets land near each
    other, which is enough to verify vector store contracts without any service.
    """

    name = "hash"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model = f"sha256-{dim}"

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
