"""Migrate every chunk from ChromaDB into Milvus.

The existing embeddings are copied as-is, so the migration costs nothing in API
spend and guarantees that any retrieval difference comes from the index, not
from re-embedded text.

This script only ever inserts. It cannot recognise a chunk it has already
written, because the Milvus primary key is auto-generated rather than derived
from the chunk key. Running it twice therefore duplicates rows. To keep that
from happening silently, it refuses to run against a non-empty collection
unless you pass --recreate.

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
        help=(
            "Drop the Milvus collection first. Required after changing the schema or dim, "
            "and after rebuilding the Chroma baseline with --reset."
        ),
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
    else:
        existing = target.count()
        if existing:
            print(
                f"Milvus already holds {existing} chunks and this script only inserts. "
                "Migrating now would leave those rows behind as duplicates or stale "
                "content, and every retrieval number measured afterwards would describe "
                "a polluted collection rather than the baseline. Re-run with --recreate "
                "to drop the collection first.",
                file=sys.stderr,
            )
            return 1

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
