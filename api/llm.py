"""Chat model providers, mirroring the embedding provider split.

Phase 2 consumes this module; it lives here now so that both providers stay in
step and the Phase 1 configuration already describes the whole system.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx

from api.config import Provider, settings

Message = dict[str, str]


class ChatModel(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        """Return the full answer as a single string."""

    @abstractmethod
    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        """Yield answer fragments as they are produced."""

    def describe(self) -> str:
        return f"{self.name}:{self.model}"

    def check_ready(self) -> str:
        """Validate lightweight provider readiness and return a probe detail."""
        return self.describe()


class OpenAIChat(ChatModel):
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "The 'openai' package is not installed. Run `uv sync --extra openai`, "
                "or set LLM_PROVIDER=ollama to use local models instead."
            ) from exc

        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Set it in .env, or set LLM_PROVIDER=ollama."
            )
        self.model = model or settings.openai_llm_model
        url = base_url or settings.openai_base_url
        self._client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)

    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaChat(ChatModel):
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or settings.ollama_llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout or settings.ollama_timeout_seconds

    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            response = client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")

    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    fragment = event.get("message", {}).get("content", "")
                    if fragment:
                        yield fragment
                    if event.get("done"):
                        return

    def check_ready(self) -> str:
        """Confirm that Ollama is reachable and the configured chat model exists."""
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            response = client.get("/api/tags")
            response.raise_for_status()

        models = response.json().get("models", [])
        available = {
            value
            for item in models
            for value in (item.get("name"), item.get("model"))
            if isinstance(value, str) and value
        }
        aliases = {self.model}
        if self.model.endswith(":latest"):
            aliases.add(self.model.removesuffix(":latest"))
        elif ":" not in self.model:
            aliases.add(f"{self.model}:latest")

        if aliases.isdisjoint(available):
            listed = ", ".join(sorted(available)) or "none"
            raise RuntimeError(
                f"Ollama model '{self.model}' is unavailable; available models: {listed}"
            )
        return self.describe()


def get_llm(provider: Provider | None = None, model: str | None = None) -> ChatModel:
    provider = provider or settings.llm_provider
    if provider == "openai":
        return OpenAIChat(model=model)
    if provider == "ollama":
        return OllamaChat(model=model)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'openai' or 'ollama'.")
