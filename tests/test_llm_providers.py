"""Offline tests for runtime chat-provider selection."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.llm import ChatModel, DeepSeekChat, Message, OllamaChat, OpenAIChat, get_llm
from api.main import app
from api.schemas import QueryRequest


class ReadyChat(ChatModel):
    name = "deepseek"
    model = "deepseek-chat"

    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        return "ok"

    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        yield "ok"


def test_query_request_keeps_provider_override_optional() -> None:
    payload = QueryRequest(query="hello")
    assert payload.llm_provider is None
    assert payload.llm_model is None


def test_query_request_accepts_supported_provider() -> None:
    payload = QueryRequest(
        query="hello", llm_provider="deepseek", llm_model="deepseek-chat"
    )
    assert payload.llm_provider == "deepseek"
    assert payload.llm_model == "deepseek-chat"


def test_query_request_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        QueryRequest(query="hello", llm_provider="unknown")


def test_factory_builds_all_provider_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.llm.settings.openai_api_key", "test-openai")
    monkeypatch.setattr("api.llm.settings.deepseek_api_key", "test-deepseek")
    assert isinstance(get_llm("ollama", "qwen2.5:3b"), OllamaChat)
    assert isinstance(get_llm("openai", "gpt-4o-mini"), OpenAIChat)
    assert isinstance(get_llm("deepseek", "deepseek-chat"), DeepSeekChat)


def test_hosted_provider_requires_server_side_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.llm.settings.deepseek_api_key", "")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_llm("deepseek", "deepseek-chat")


def test_llm_check_returns_sanitized_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routers.llm.get_llm", lambda provider, model: ReadyChat())
    with TestClient(app) as client:
        response = client.post(
            "/llm/check", json={"provider": "deepseek", "model": "deepseek-chat"}
        )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "provider": "deepseek",
        "model": "deepseek-chat",
        "detail": "deepseek:deepseek-chat",
    }
