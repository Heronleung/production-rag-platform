"""The vector store abstraction.

Every caller in this codebase depends on ``VectorStore`` and never on a concrete
client. Swapping ChromaDB for Milvus is therefore a configuration change rather
than a rewrite, and the same contract test suite can be run against any backend.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """One embedded unit of text, ready to be written to a vector store."""

    text: str
    source: str
    chunk_index: int
    vector: list[float] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def key(self) -> str:
        """Stable identity of a chunk, used for deduplication and for comparing
        results across backends."""
        return f"{self.source}::{self.chunk_index}"


@dataclass(slots=True)
class SearchHit:
    """One retrieval result.

    ``score`` is always *higher is better* and normalised to a similarity, even
    when the underlying backend reports a distance.

    ``vector`` is populated only when the caller passes ``return_vectors=True``
    to :meth:`VectorStore.search`. It is required by the MMR algorithm and is
    left empty otherwise to avoid unnecessary data transfer.
    """

    text: str
    source: str
    chunk_index: int
    score: float
    vector: list[float] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.source}::{self.chunk_index}"


class VectorStore(ABC):
    """Minimal contract that every backend must satisfy."""

    dim: int

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int:
        """Write chunks and return the number of rows persisted."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
        return_vectors: bool = False,
    ) -> list[SearchHit]:
        """Return the ``top_k`` most similar chunks, best match first.

        When ``return_vectors`` is ``True``, each :class:`SearchHit` will have
        its ``vector`` field populated. This is needed by the MMR algorithm and
        avoids a second embedding round-trip.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of chunks currently stored."""

    @abstractmethod
    def drop(self) -> None:
        """Delete the whole collection. Used by tests and by re-migration."""

    def _validate(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if len(chunk.vector) != self.dim:
                raise ValueError(
                    f"vector dimension mismatch for {chunk.key}: "
                    f"got {len(chunk.vector)}, store expects {self.dim}"
                )
