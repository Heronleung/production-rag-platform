"""Rebuild a persistent ChromaDB baseline for the Phase 1 parity benchmark.

Why this script exists
----------------------
The predecessor project (``RAG-Chatbot-Project-Ollama3.2``) built its index like
this::

    vectordb = Chroma.from_documents(chunks, embedding_model)

No ``persist_directory`` was passed, so Chroma ran in ephemeral in-memory mode:
the index was rebuilt from the uploaded PDF on every request and discarded when
the process exited. There is therefore **no stored Chroma data to migrate** -
``scripts/migrate_chroma_to_milvus.py`` has nothing to read.

That also breaks the Phase 1 acceptance criterion, which compares top-k results
between Chroma and Milvus. A comparison is only meaningful when both sides were
built from the same text, the same chunking and the same embedder; otherwise the
overlap number measures the difference between two embedding models rather than
the difference between two vector stores.

This script rebuilds that baseline reproducibly:

1. Load documents from a directory (``.pdf``, ``.md``, ``.txt``).
2. Split them with the same parameters the old project used
   (chunk size 1000, overlap 50, recursive separators).
3. Embed them with whatever ``EMBEDDING_PROVIDER``/model is configured now.
4. Write them into a **persistent** Chroma collection.

After this, ``migrate_chroma_to_milvus.py`` has real data to move, and
``benchmark_overlap.py`` compares like with like.

The chunker itself now lives in :mod:`ingestion.chunking` (Phase 2), so this
script and ``POST /ingest`` split text identically instead of keeping two copies
of the same code.

Usage::

    uv run python scripts/build_chroma_baseline.py --input ./data
    uv run python scripts/build_chroma_baseline.py --input ./data --reset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from api.config import settings
from api.embeddings import get_embedder
from api.vectorstore.base import Chunk
from api.vectorstore.chroma_store import ChromaStore
from ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SUPPORTED_SUFFIXES,
    decode_text_bytes,
    read_pdf_bytes,
    split_text,
)


def read_document(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        try:
            return read_pdf_bytes(data)
        except RuntimeError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(str(exc)) from exc
    return decode_text_bytes(data)


def collect_documents(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise SystemExit(
            f"No supported documents found under {input_dir}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data"),
        help="Directory containing the source documents (default: ./data)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="How many chunks to embed per call (default: 64)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing Chroma collection before writing",
    )
    args = parser.parse_args()

    embedder = get_embedder()
    store = ChromaStore()

    print(f"Provider          : {settings.embedding_provider}")
    print(f"Embedding model   : {embedder.model}")
    print(f"Dimension         : {embedder.dim}")
    print(f"Chroma path       : {settings.chroma_path}")
    print(f"Chroma collection : {store.collection_name}")
    print()

    if args.reset:
        store.drop()
        print("Existing collection dropped.")

    paths = collect_documents(args.input)
    print(f"Found {len(paths)} document(s) under {args.input}")

    started = time.perf_counter()
    total_chunks = 0

    for path in paths:
        text = read_document(path)
        pieces = split_text(text, args.chunk_size, args.chunk_overlap)
        if not pieces:
            print(f"  {path.name}: no extractable text, skipped")
            continue

        source = str(path.relative_to(args.input))
        for start in range(0, len(pieces), args.batch_size):
            window = pieces[start : start + args.batch_size]
            vectors = embedder.embed(window)
            store.upsert(
                [
                    Chunk(
                        text=piece,
                        source=source,
                        chunk_index=start + offset,
                        vector=vector,
                    )
                    for offset, (piece, vector) in enumerate(zip(window, vectors, strict=True))
                ]
            )
        total_chunks += len(pieces)
        print(f"  {path.name}: {len(pieces)} chunk(s)")

    elapsed = time.perf_counter() - started
    stored = store.count()

    print()
    print(f"Chunks written    : {total_chunks}")
    print(f"Collection count  : {stored}")
    print(f"Elapsed           : {elapsed:.1f}s")

    if stored != total_chunks:
        print(
            "\nWARNING: collection count does not match the number of chunks written. "
            "If the collection already held data, re-run with --reset."
        )
        return 1

    print("\nOK: baseline built. Next: scripts/migrate_chroma_to_milvus.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
