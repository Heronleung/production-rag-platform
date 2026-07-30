"""Milvus implementation of :class:`VectorStore`.

Things that are easy to get wrong and are handled explicitly here:

1. A collection must be **loaded into memory** before any search succeeds.
   Inserting alone is not enough; a fresh collection returns zero results.
2. Index building is **asynchronous**. Benchmarking before the index is ready
   measures a brute-force scan and produces misleading latency numbers, so
   :meth:`wait_for_index` is provided and called after migration.
3. The schema is fixed at creation time. Changing ``dim`` later requires
   dropping and rebuilding the collection.
"""

from __future__ import annotations

import time

from pymilvus import DataType, MilvusClient

from api.config import settings
from api.vectorstore.base import Chunk, SearchHit, VectorStore

_BASE_OUTPUT_FIELDS = ["text", "source", "chunk_index"]


class MilvusStore(VectorStore):
    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
        token: str | None = None,
        metric_type: str | None = None,
        hnsw_m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
    ) -> None:
        self.collection = collection or settings.milvus_collection
        self.dim = dim or settings.embedding_dim
        self.metric_type = metric_type or settings.milvus_metric_type
        self.hnsw_m = hnsw_m or settings.milvus_hnsw_m
        self.ef_construction = ef_construction or settings.milvus_hnsw_ef_construction
        self.ef_search = ef_search or settings.milvus_hnsw_ef_search

        self.client = MilvusClient(
            uri=uri or settings.milvus_uri,
            token=token if token is not None else settings.milvus_token,
        )

        if not self.client.has_collection(self.collection):
            self._create_collection()
        self.client.load_collection(self.collection)

    # ------------------------------------------------------------------ setup

    def _create_collection(self) -> None:
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("created_at", DataType.INT64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type=self.metric_type,
            params={"M": self.hnsw_m, "efConstruction": self.ef_construction},
        )
        # A scalar index on source keeps filtered search cheap once the
        # collection grows past a few hundred thousand rows.
        index_params.add_index(field_name="source", index_type="INVERTED")

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def wait_for_index(self, timeout: float = 300.0, poll_interval: float = 2.0) -> None:
        """Block until the vector index finishes building."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = self.client.describe_index(self.collection, index_name="vector")
            except Exception:  # noqa: BLE001 - index not registered yet
                time.sleep(poll_interval)
                continue
            if str(state.get("state", "")).lower() in {"finished", "complete", "completed"}:
                return
            time.sleep(poll_interval)
        raise TimeoutError(f"index on '{self.collection}' did not finish within {timeout}s")

    # ------------------------------------------------------------------ write

    def upsert(self, chunks: list[Chunk], batch_size: int = 1000) -> int:
        if not chunks:
            return 0
        self._validate(chunks)

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            rows = [
                {
                    "vector": chunk.vector,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "created_at": chunk.created_at,
                }
                for chunk in batch
            ]
            self.client.insert(collection_name=self.collection, data=rows)
            written += len(rows)

        self.client.flush(self.collection)
        return written

    # ------------------------------------------------------------------- read

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
        return_vectors: bool = False,
    ) -> list[SearchHit]:
        output_fields = list(_BASE_OUTPUT_FIELDS)
        if return_vectors:
            # Include the stored vector so the caller can run MMR without a
            # second embedding round-trip.
            output_fields.append("vector")

        expr = f'source == "{source_filter}"' if source_filter else ""
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            filter=expr,
            output_fields=output_fields,
            search_params={"metric_type": self.metric_type, "params": {"ef": self.ef_search}},
        )
        if not results:
            return []
        return [self._to_hit(raw, return_vectors=return_vectors) for raw in results[0]]

    def _to_hit(self, raw: dict, return_vectors: bool = False) -> SearchHit:
        entity = raw.get("entity", raw)
        distance = float(raw.get("distance", 0.0))
        # COSINE and IP already report similarity; L2 reports a distance.
        score = 1.0 / (1.0 + distance) if self.metric_type == "L2" else distance
        return SearchHit(
            text=entity.get("text", ""),
            source=entity.get("source", ""),
            chunk_index=int(entity.get("chunk_index", -1)),
            score=score,
            vector=list(entity.get("vector", [])) if return_vectors else [],
        )

    def count(self) -> int:
        rows = self.client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["count(*)"],
        )
        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
