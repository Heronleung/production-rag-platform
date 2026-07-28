"""Migrate every chunk from ChromaDB into Milvus.

The existing embeddings are copied as-is, so the migration costs nothing in API
spend and guarantees that any retrieval difference comes from the index, not
from re-embedded text.

Usage:
    uv run python scripts/migrate_chroma_to_milvus.py --batch-size 1000
    uv run python scripts/migrate_chroma_to_milvus.py --recreate
"""

from __future__ import annotations

import argparse
import sys

from api.vectorstore.chroma_store import ChromaStore
from api.vectorstore.milvus_store import MilvusStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy Chroma chunks into Milvus.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the Milvus collection first. Required after changing the schema or dim.",
    )
    parser.add_argument(
        "--skip-index-wait",
        action="store_true",
        help="Do not block until the HNSW index finishes building.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = ChromaStore()
    expected = source.count()
    if expected == 0:
        print("Chroma collection is empty, nothing to migrate.")
        return 1
    print(f"Chroma reports {expected} chunks.")

    target = MilvusStore()
    if args.recreate:
        print("Dropping existing Milvus collection.")
        target.drop()
        target = MilvusStore()

    buffer: list = []
    written = 0
    for chunk in source.iter_chunks(batch_size=args.batch_size):
        buffer.append(chunk)
        if len(buffer) >= args.batch_size:
            written += target.upsert(buffer)
            buffer.clear()
            print(f"  migrated {written}/{expected}", flush=True)
    if buffer:
        written += target.upsert(buffer)

    if not args.skip_index_wait:
        print("Waiting for the HNSW index to finish building.")
        target.wait_for_index()

    target.client.load_collection(target.collection)
    actual = target.count()
    print(f"Milvus now reports {actual} chunks (wrote {written}).")

    if actual != expected:
        print(f"MISMATCH: expected {expected}, found {actual}.", file=sys.stderr)
        return 1
    print("Row counts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
