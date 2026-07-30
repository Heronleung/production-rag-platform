"""Unit tests for the shared chunker.

The splitter is inherited unchanged from Phase 1 (``build_chroma_baseline.py``);
these tests pin its behaviour now that ``POST /ingest`` depends on it too, so a
Phase 3 rewrite of the chunking strategy cannot silently change what the API
stores.
"""

from __future__ import annotations

import pytest

from ingestion.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("a short paragraph", chunk_size=100, chunk_overlap=10) == [
        "a short paragraph"
    ]


def test_blank_text_produces_no_chunks() -> None:
    assert split_text("   \n\n  ", chunk_size=100, chunk_overlap=10) == []


def test_defaults_match_the_phase_1_baseline() -> None:
    assert (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP) == (1000, 50)


def test_long_text_is_split_and_bounded() -> None:
    chunk_size, overlap = 200, 20
    text = "\n\n".join(f"Paragraph {index} " + "word " * 60 for index in range(10))
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=overlap)

    assert len(chunks) > 1
    # A chunk is at most one window plus the overlap prefix copied from the
    # previous window.
    assert all(len(chunk) <= chunk_size + overlap for chunk in chunks)


def test_overlap_prefixes_the_previous_tail() -> None:
    # A uniform body makes the window boundaries deterministic: 250 characters
    # split into 100 + 100 + 50, then 10 characters of overlap prepended to
    # every chunk after the first.
    assert [len(chunk) for chunk in split_text("x" * 250, chunk_size=100, chunk_overlap=10)] == [
        100,
        110,
        60,
    ]


def test_no_separator_available_hard_cuts() -> None:
    chunks = split_text("x" * 250, chunk_size=100, chunk_overlap=0)
    assert [len(chunk) for chunk in chunks] == [100, 100, 50]


def test_paragraph_boundaries_are_preferred() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = split_text(text, chunk_size=20, chunk_overlap=0)
    assert chunks == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_all_source_text_survives_chunking() -> None:
    text = "\n\n".join(f"Sentence {index} about Milvus indexes." for index in range(40))
    joined = "".join(split_text(text, chunk_size=120, chunk_overlap=0))
    for index in (0, 17, 39):
        assert f"Sentence {index} about Milvus indexes." in joined


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        split_text("text", chunk_size=100, chunk_overlap=100)
