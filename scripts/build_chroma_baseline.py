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

# The old project's values, kept identical on purpose so the baseline is a fair
# stand-in for what the predecessor system actually retrieved.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """A recursive character splitter equivalent to LangChain's default.

    Reimplemented here rather than pulled in as a dependency: the behaviour is
    small enough to own, and Phase 3 will replace it with several configurable
    strategies anyway.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _split(segment: str, separators: list[str]) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment] if segment.strip() else []
        if not separators:
            # No separator left: hard-cut the segment.
            return [segment[i : i + chunk_size] for i in range(0, len(segment), chunk_size)]

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return [segment[i : i + chunk_size] for i in range(0, len(segment), chunk_size)]

        pieces = segment.split(separator)
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            candidate = piece if not buffer else buffer + separator + piece
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue
            if buffer:
                merged.append(buffer)
            if len(piece) > chunk_size:
                merged.extend(_split(piece, rest))
                buffer = ""
            else:
                buffer = piece
        if buffer:
            merged.append(buffer)
        return [item for item in merged if item.strip()]

    raw = _split(text, SEPARATORS)

    # Apply the overlap by prefixing each chunk with the tail of the previous one.
    if chunk_overlap <= 0 or len(raw) < 2:
        return raw
    overlapped = [raw[0]]
    for previous, current in zip(raw, raw[1:], strict=False):
        overlapped.append(previous[-chunk_overlap:] + current)
    return overlapped


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(
                "Reading PDFs requires pypdf. Install it with: uv add pypdf"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


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
