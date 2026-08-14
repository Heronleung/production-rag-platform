"""Chat model providers with a shared streaming contract."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx

from api.config import LLMProvider, settings

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
        return self.describe()


class OpenAICompatibleChat(ChatModel):
    """HTTP client shared by OpenAI and DeepSeek without exposing API keys."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
    ) -> None:
        if not api_key:
            env_name = "OPENAI_API_KEY" if name == "openai" else "DEEPSEEK_API_KEY"
            raise ValueError(f"{env_name} is empty; configure it on the backend server.")
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        status_code = response.status_code
        if status_code in {401, 403}:
            detail = "authentication failed"
        elif status_code == 404:
            detail = f"model '{self.model}' or endpoint was not found"
        elif status_code == 429:
            detail = "rate limit or quota exceeded"
        else:
            detail = f"upstream returned HTTP {status_code}"
        raise RuntimeError(f"{self.name} {detail}")

    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            response = client.post("/chat/completions", headers=self._headers, json=payload)
        self._raise_for_status(response)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            with client.stream(
                "POST", "/chat/completions", headers=self._headers, json=payload
            ) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        return
                    event = json.loads(data)
                    fragment = event.get("choices", [{}])[0].get("delta", {}).get("content")
                    if fragment:
                        yield fragment

    def check_ready(self) -> str:
        with httpx.Client(base_url=self.base_url, timeout=self._timeout) as client:
            response = client.get("/models", headers=self._headers)
        self._raise_for_status(response)
        model_ids = {
            item.get("id")
            for item in response.json().get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if model_ids and self.model not in model_ids:
            raise RuntimeError(f"{self.name} model '{self.model}' is unavailable")
        return self.describe()


class OpenAIChat(OpenAICompatibleChat):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            name="openai",
            model=model or settings.openai_llm_model,
            api_key=settings.openai_api_key if api_key is None else api_key,
            base_url=base_url or settings.openai_base_url,
            timeout=settings.hosted_llm_timeout_seconds,
        )


class DeepSeekChat(OpenAICompatibleChat):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            name="deepseek",
            model=model or settings.deepseek_llm_model,
            api_key=settings.deepseek_api_key if api_key is None else api_key,
            base_url=base_url or settings.deepseek_base_url,
            timeout=settings.hosted_llm_timeout_seconds,
        )


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


def get_llm(provider: LLMProvider | None = None, model: str | None = None) -> ChatModel:
    selected = provider or settings.llm_provider
    if selected == "openai":
        return OpenAIChat(model=model)
    if selected == "deepseek":
        return DeepSeekChat(model=model)
    if selected == "ollama":
        return OllamaChat(model=model)
    raise ValueError(
        f"Unknown LLM provider: {selected!r}. Use 'openai', 'deepseek', or 'ollama'."
    )
