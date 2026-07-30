"""Golden-dataset loading and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from evaluation.schema import GoldenSample


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path: str | Path) -> list[GoldenSample]:
    """Load JSONL samples and reject malformed rows or duplicate ids."""
    source = Path(path)
    samples: list[GoldenSample] = []
    ids: set[str] = set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            sample = GoldenSample.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{source}:{line_number}: invalid golden sample: {exc}") from exc
        if sample.id in ids:
            raise ValueError(f"{source}:{line_number}: duplicate sample id {sample.id!r}")
        ids.add(sample.id)
        samples.append(sample)
    if not samples:
        raise ValueError(f"{source}: dataset is empty")
    return samples
