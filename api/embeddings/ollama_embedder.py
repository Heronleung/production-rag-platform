"""Ollama embedding provider: local models, no API key, no per-token cost.

Setup:

    ollama serve
    ollama pull nomic-embed-text

Or use the bundled container:

    docker compose -f deploy/compose/ollama.yml up -d
    docker exec rag-ollama ollama pull nomic-embed-text

Two endpoints exist across Ollama versions. ``/api/embed`` (0.1.39 and newer)
accepts a batch; older builds only expose ``/api/embeddings``, which handles a
single prompt at a time. This client prefers the batch endpoint and falls back
automatically, so it works on both.
"""

from __future__ import annotations

import httpx

from api.config import settings
from api.embeddings.base import Embedder


class OllamaConnectionError(RuntimeError):
    pass


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or settings.ollama_embedding_model
        self.dim = dim or settings.embedding_dim
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout or settings.ollama_timeout_seconds,
        )
        self._use_legacy_endpoint = False

    # ------------------------------------------------------------------ public

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._use_legacy_endpoint:
            return [self._embed_legacy(text) for text in texts]
        try:
            return self._embed_batch(texts)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            # Older Ollama build: batch endpoint does not exist.
            self._use_legacy_endpoint = True
            return [self._embed_legacy(text) for text in texts]

    def is_available(self) -> bool:
        """Return True when the daemon answers and the model is pulled."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {entry.get("name", "") for entry in response.json().get("models", [])}
        base_names = {name.split(":")[0] for name in names}
        return self.model in names or self.model.split(":")[0] in base_names

    def close(self) -> None:
        self._client.close()

    # ----------------------------------------------------------------- private

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        data = self._post("/api/embed", payload)
        vectors = data.get("embeddings")
        if not vectors:
            raise RuntimeError(f"Ollama returned no embeddings for model '{self.model}'.")
        return [list(map(float, vector)) for vector in vectors]

    def _embed_legacy(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        data = self._post("/api/embeddings", payload)
        vector = data.get("embedding")
        if not vector:
            raise RuntimeError(f"Ollama returned no embedding for model '{self.model}'.")
        return list(map(float, vector))

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self._client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`, "
                f"or run `docker compose -f deploy/compose/ollama.yml up -d`."
            ) from exc
        if response.status_code == 404 and "model" in response.text.lower():
            raise RuntimeError(
                f"Model '{self.model}' is not pulled. Run `ollama pull {self.model}`."
            )
        response.raise_for_status()
        return response.json()
