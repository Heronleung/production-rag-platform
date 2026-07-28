"""Pure-Python reference implementation.

Its only job is to define the expected behaviour of the contract in code, so the
shared test suite has a backend that always passes and needs no services.
"""

from __future__ import annotations

import math

from api.vectorstore.base import Chunk, SearchHit, VectorStore


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryStore(VectorStore):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._rows: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> int:
        self._validate(chunks)
        for chunk in chunks:
            self._rows[chunk.key] = chunk
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        candidates = [
            chunk
            for chunk in self._rows.values()
            if source_filter is None or chunk.source == source_filter
        ]
        scored = [
            SearchHit(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=cosine_similarity(query_vector, chunk.vector),
            )
            for chunk in candidates
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._rows)

    def drop(self) -> None:
        self._rows.clear()
