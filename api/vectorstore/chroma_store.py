"""ChromaDB implementation of :class:`VectorStore`.

This backend is retained for two reasons only: it is the migration source, and
it is the baseline the Milvus results are compared against in Phase 1.
"""

from __future__ import annotations

from api.config import settings
from api.vectorstore.base import Chunk, SearchHit, VectorStore


class ChromaStore(VectorStore):
    def __init__(
        self,
        path: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
    ) -> None:
        import chromadb

        self.dim = dim or settings.embedding_dim
        self.collection_name = collection or settings.chroma_collection
        self._client = chromadb.PersistentClient(path=path or settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._validate(chunks)
        self._collection.upsert(
            ids=[chunk.key for chunk in chunks],
            embeddings=[chunk.vector for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "created_at": chunk.created_at,
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"source": source_filter} if source_filter else None,
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            metadata = metadata or {}
            hits.append(
                SearchHit(
                    text=text or "",
                    source=str(metadata.get("source", "")),
                    chunk_index=int(metadata.get("chunk_index", -1)),
                    # Chroma returns cosine *distance*; convert to similarity.
                    score=1.0 - float(distance),
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def drop(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ----------------------------------------------------- migration helper

    def iter_chunks(self, batch_size: int = 1000):
        """Yield every stored chunk, including its existing embedding.

        Reusing the stored embeddings is what makes the migration free: no text
        is sent to the embedding API a second time.

        Note on the ``is None`` checks below: chromadb returns ``embeddings`` as a
        numpy ndarray, not a list. Writing ``page.get("embeddings") or []`` looks
        harmless but evaluates the array in a boolean context, which numpy
        refuses with "truth value of an array with more than one element is
        ambiguous". Any ``or``/``if x``/``not x`` shortcut applied to a value that
        may be an ndarray is a bug waiting to happen; compare against ``None``
        explicitly instead.
        """
        offset = 0
        total = self.count()
        while offset < total:
            page = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )

            ids = page.get("ids")
            if ids is None or len(ids) == 0:
                return

            embeddings = page.get("embeddings")
            documents = page.get("documents")
            metadatas = page.get("metadatas")

            # Chroma may omit a requested key entirely; fall back to per-row
            # placeholders so the zip below stays aligned with ``ids``.
            if embeddings is None:
                embeddings = [None] * len(ids)
            if documents is None:
                documents = [None] * len(ids)
            if metadatas is None:
                metadatas = [None] * len(ids)

            for row_id, embedding, document, metadata in zip(
                ids, embeddings, documents, metadatas, strict=False
            ):
                if embedding is None or len(embedding) == 0:
                    # A row without a vector cannot be migrated; skipping it is
                    # safer than inserting a chunk that can never be retrieved.
                    # The count check in the migration script will flag the gap.
                    print(f"  skipped {row_id}: no stored embedding")
                    continue
                metadata = metadata or {}
                yield Chunk(
                    text=document or "",
                    source=str(metadata.get("source", "unknown")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    vector=[float(value) for value in embedding],
                )

            offset += len(ids)
