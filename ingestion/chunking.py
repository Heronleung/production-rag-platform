"""Document parsing and chunking, shared by the ingestion API and offline scripts.

Extracted from ``scripts/build_chroma_baseline.py`` (Phase 1) so that
``POST /ingest`` (Phase 2) and the Chroma baseline builder chunk text
identically instead of maintaining two copies of the same splitter.
"""

from __future__ import annotations

import io

# The predecessor project's values, kept identical on purpose so that ingested
# chunks stay comparable with the Phase 1 parity benchmark.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
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


def read_pdf_bytes(data: bytes) -> str:
    """Extract text from in-memory PDF bytes.

    Used by ``POST /ingest``, which receives an uploaded file in memory and
    should not have to write a temporary file just to parse it.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Reading PDFs requires pypdf. Install it with: uv add pypdf"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def decode_text_bytes(data: bytes) -> str:
    """Decode an uploaded ``.md``/``.txt`` file, tolerating bad bytes."""
    return data.decode("utf-8", errors="replace")
