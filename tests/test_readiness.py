"""Focused readiness regression tests for dependency and model availability."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient

from api.dependencies import get_chat_model, get_embedder_singleton, get_vector_store
from api.embeddings import HashEmbedder
from api.llm import ChatModel, Message
from api.main import app
from api.vectorstore.memory_store import InMemoryStore


class ReadyChat(ChatModel):
    name = "fake"
    model = "ready-model"

    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        return "ready"

    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        yield "ready"


class MissingChat(ReadyChat):
    model = "missing-model"

    def check_ready(self) -> str:
        raise RuntimeError("configured chat model is unavailable")


def _set_overrides(chat: ChatModel) -> None:
    app.dependency_overrides[get_embedder_singleton] = lambda: HashEmbedder(dim=64)
    app.dependency_overrides[get_vector_store] = lambda: InMemoryStore(dim=64)
    app.dependency_overrides[get_chat_model] = lambda: chat


def test_readyz_reports_chat_model_when_available() -> None:
    _set_overrides(ReadyChat())
    try:
        with TestClient(app) as client:
            response = client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        dependencies = {item["name"]: item for item in body["dependencies"]}
        assert dependencies["chat_model"] == {
            "name": "chat_model",
            "ok": True,
            "detail": "fake:ready-model",
        }
    finally:
        app.dependency_overrides.clear()


def test_readyz_returns_503_when_chat_model_is_missing() -> None:
    _set_overrides(MissingChat())
    try:
        with TestClient(app) as client:
            response = client.get("/readyz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        dependencies = {item["name"]: item for item in body["dependencies"]}
        assert dependencies["chat_model"]["ok"] is False
        assert "unavailable" in dependencies["chat_model"]["detail"]
    finally:
        app.dependency_overrides.clear()
