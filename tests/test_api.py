"""HTTP contract tests for the Phase 2 API.

No Milvus and no Ollama are required: the three dependency providers are
overridden with the offline ``HashEmbedder``, an ``InMemoryStore`` and a fake
chat model. That is the whole point of routing them through
:mod:`api.dependencies` - the full request path (validation, middleware, error
handling, SSE framing) is exercised, only the external services are replaced.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_chat_model, get_embedder_singleton, get_vector_store
from api.embeddings import HashEmbedder
from api.llm import ChatModel, Message
from api.main import app
from api.vectorstore.memory_store import InMemoryStore

EMBEDDING_DIM = 64


class FakeChat(ChatModel):
    """Echoes a fixed answer and records the prompt it was given."""

    name = "fake"

    def __init__(self, answer: str = "Milvus supports HNSW [1].") -> None:
        self.model = "fake-1"
        self.answer = answer
        self.last_messages: list[Message] = []

    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        self.last_messages = messages
        return self.answer

    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        self.last_messages = messages
        for word in self.answer.split(" "):
            yield word + " "


class BrokenChat(FakeChat):
    def stream(self, messages: list[Message], temperature: float = 0.0) -> Iterator[str]:
        yield "partial "
        raise RuntimeError("upstream model died")


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(dim=EMBEDDING_DIM)


@pytest.fixture
def chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def client(store: InMemoryStore, chat: FakeChat) -> Iterator[TestClient]:
    embedder = HashEmbedder(dim=EMBEDDING_DIM)
    app.dependency_overrides[get_embedder_singleton] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_chat_model] = lambda: chat
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _ingest(client: TestClient, name: str = "milvus.md", body: str | None = None):
    text = body or (
        "Milvus supports HNSW and IVF_FLAT indexes.\n\n"
        "A collection must be loaded into memory before it can be searched.\n\n"
        "RAGAS measures faithfulness and answer relevancy."
    )
    return client.post("/ingest", files={"file": (name, text.encode("utf-8"), "text/markdown")})


def _sse_events(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        name = ""
        payload = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = line.removeprefix("data: ")
        if name:
            events.append((name, json.loads(payload) if payload else {}))
    return events


# ----------------------------------------------------------------- basics


def test_root_reports_service(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["service"] == "production-rag-platform"


def test_healthz_never_touches_dependencies(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_id_is_echoed_back(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    assert client.get("/healthz").headers.get("X-Request-ID")


def test_readyz_reports_dependencies(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {dep["name"] for dep in body["dependencies"]} == {"embedder", "vector_store"}


def test_openapi_documents_both_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/ingest" in paths
    assert "/query" in paths


# ----------------------------------------------------------------- ingest


def test_ingest_writes_chunks(client: TestClient, store: InMemoryStore) -> None:
    response = _ingest(client)
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "milvus.md"
    assert body["chunks_written"] >= 1
    assert store.count() == body["chunks_written"]


def test_ingest_strips_client_paths(client: TestClient) -> None:
    response = _ingest(client, name="../../etc/passwd.md")
    assert response.status_code == 201
    assert response.json()["source"] == "passwd.md"


def test_ingest_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("notes.docx", b"data", "application/x")})
    assert response.status_code == 415


def test_ingest_rejects_empty_file(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_ingest_rejects_text_without_content(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("blank.txt", b"   \n  ", "text/plain")})
    assert response.status_code == 422


def test_ingest_rejects_bad_chunk_params(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        params={"chunk_size": 200, "chunk_overlap": 200},
        files={"file": ("a.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_error_responses_carry_request_id(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("notes.docx", b"data", "application/x")})
    assert response.json()["request_id"]


# ------------------------------------------------------------------ query


def test_query_non_streaming_returns_answer_and_citations(
    client: TestClient, chat: FakeChat
) -> None:
    _ingest(client)
    response = client.post("/query", json={"query": "Which indexes does Milvus support?",
                                           "stream": False, "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == chat.answer
    assert 1 <= len(body["citations"]) <= 2
    assert body["llm_model"] == "fake:fake-1"


def test_query_prompt_contains_numbered_context(client: TestClient, chat: FakeChat) -> None:
    _ingest(client)
    client.post("/query", json={"query": "Milvus indexes", "stream": False})
    prompt = chat.last_messages[-1]["content"]
    assert "[1]" in prompt
    assert "Question: Milvus indexes" in prompt


def test_query_source_filter_is_applied(client: TestClient) -> None:
    _ingest(client, name="milvus.md")
    _ingest(client, name="k8s.md", body="Kubernetes autoscaling reacts to custom metrics.")
    response = client.post(
        "/query",
        json={"query": "Milvus indexes", "stream": False, "source_filter": "k8s.md"},
    )
    assert response.status_code == 200
    assert {c["source"] for c in response.json()["citations"]} == {"k8s.md"}


def test_query_without_corpus_returns_404(client: TestClient) -> None:
    response = client.post("/query", json={"query": "anything", "stream": False})
    assert response.status_code == 404


def test_query_validates_input(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x", "top_k": 0}).status_code == 422


def test_query_streams_sse_events(client: TestClient, chat: FakeChat) -> None:
    _ingest(client)
    response = client.post("/query", json={"query": "Milvus indexes", "top_k": 2})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "citations"
    assert names[-1] == "done"
    assert "token" in names

    streamed = "".join(data["text"] for name, data in events if name == "token")
    assert streamed.strip() == chat.answer


def test_stream_failure_is_reported_as_an_error_event(store: InMemoryStore) -> None:
    embedder = HashEmbedder(dim=EMBEDDING_DIM)
    app.dependency_overrides[get_embedder_singleton] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_chat_model] = lambda: BrokenChat()
    try:
        with TestClient(app) as client:
            _ingest(client)
            response = client.post("/query", json={"query": "Milvus indexes"})
            names = [name for name, _ in _sse_events(response.text)]
            assert names[-1] == "error"
    finally:
        app.dependency_overrides.clear()
