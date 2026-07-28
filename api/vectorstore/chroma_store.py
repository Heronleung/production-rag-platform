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
        """
        offset = 0
        total = self.count()
        while offset < total:
            page = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            ids = page.get("ids") or []
            if not ids:
                return
            embeddings = page.get("embeddings") or []
            documents = page.get("documents") or []
            metadatas = page.get("metadatas") or []
            for embedding, document, metadata in zip(
                embeddings, documents, metadatas, strict=False
            ):
                metadata = metadata or {}
                yield Chunk(
                    text=document or "",
                    source=str(metadata.get("source", "unknown")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    vector=list(embedding),
                )
            offset += len(ids)
