"""Verify the embedding provider before touching Milvus.

This exists because the Milvus collection schema is immutable: if the configured
dimension does not match the model, the mistake only surfaces at insert time and
the collection has to be dropped and rebuilt. Run this first.

Usage:
    uv run python scripts/check_embedder.py
    uv run python scripts/check_embedder.py --provider openai
"""

from __future__ import annotations

import argparse
import sys
import time

from api.config import settings
from api.embeddings import get_embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the configured embedder.")
    parser.add_argument("--provider", choices=["openai", "ollama"], default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provider = args.provider or settings.embedding_provider

    print(f"Provider          : {provider}")
    print(f"Configured model  : {args.model or settings.embedding_model}")
    print(f"Configured dim    : {settings.embedding_dim}")

    try:
        embedder = get_embedder(provider=provider, model=args.model)
    except Exception as exc:  # noqa: BLE001 - this script exists to report failures
        print(f"\nFAILED to build the embedder: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    try:
        actual_dim = embedder.probe_dim()
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to embed a probe string: {exc}", file=sys.stderr)
        return 1
    elapsed_ms = (time.perf_counter() - started) * 1000

    print(f"Actual dim        : {actual_dim}")
    print(f"Single-call latency: {elapsed_ms:.0f} ms")

    if actual_dim != settings.embedding_dim:
        print(
            f"\nMISMATCH: the model returns {actual_dim} dimensions but EMBEDDING_DIM is "
            f"{settings.embedding_dim}. Set EMBEDDING_DIM={actual_dim} in .env and recreate "
            f"the Milvus collection with `--recreate`.",
            file=sys.stderr,
        )
        return 1

    print("\nOK: provider reachable and dimension matches the configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
