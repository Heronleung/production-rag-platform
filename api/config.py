"""Centralised configuration, loaded from environment variables or a .env file.

The project supports two interchangeable model providers:

* ``openai`` - hosted API, needs ``OPENAI_API_KEY``.
* ``ollama`` - local models, needs a running ``ollama serve`` and no key at all.

Only ``EMBEDDING_PROVIDER`` and ``LLM_PROVIDER`` decide which one is used; no
application code branches on the provider name.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "ollama"]

# Embedding dimension is a property of the model, and the Milvus collection
# schema is immutable once created. Getting it wrong is the single most common
# Phase 1 failure, so the known values are resolved automatically.
KNOWN_EMBEDDING_DIMS: dict[tuple[str, str], int] = {
    ("openai", "text-embedding-3-small"): 1536,
    ("openai", "text-embedding-3-large"): 3072,
    ("openai", "text-embedding-ada-002"): 1536,
    ("ollama", "nomic-embed-text"): 768,
    ("ollama", "nomic-embed-text:latest"): 768,
    ("ollama", "mxbai-embed-large"): 1024,
    ("ollama", "mxbai-embed-large:latest"): 1024,
    ("ollama", "bge-m3"): 1024,
    ("ollama", "bge-m3:latest"): 1024,
    ("ollama", "snowflake-arctic-embed"): 1024,
    ("ollama", "all-minilm"): 384,
}

# Rough footprint of the local chat models this project has been run against.
# Kept here purely as documentation for anyone choosing a value for
# OLLAMA_LLM_MODEL; nothing reads this at runtime.
OLLAMA_LLM_NOTES: dict[str, str] = {
    "qwen2.5:0.5b": "~0.4GB, smoke tests only - too weak for benchmark numbers",
    "qwen2.5:1.5b": "~1.0GB, default - runs on CPU, decent Chinese, weaker citation discipline",
    "llama3.2:3b": "~2.0GB, better instruction following, still CPU-friendly",
    "llama3.1:8b": "~4.7GB, best quality of the four, wants a GPU or 16GB+ RAM",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Provider selection ---------------------------------------------
    embedding_provider: Provider = "ollama"
    llm_provider: Provider = "ollama"

    # 0 means "resolve from the table above". Set EMBEDDING_DIM explicitly only
    # when using a model this project does not know about.
    embedding_dim: int = 0

    # ---- OpenAI ----------------------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    # ---- Ollama ----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    # A small default on purpose: the local path has to stay usable on a laptop
    # with no GPU. See OLLAMA_LLM_NOTES for heavier options.
    ollama_llm_model: str = "qwen2.5:1.5b"
    ollama_timeout_seconds: float = 120.0

    # ---- Milvus ----------------------------------------------------------
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_chunks"
    milvus_metric_type: str = "COSINE"
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_hnsw_ef_search: int = 64

    # ---- Chroma (migration source only) ----------------------------------
    chroma_path: str = "./.chroma"
    chroma_collection: str = "rag_chunks"

    # ---------------------------------------------------------------- derived

    @property
    def embedding_model(self) -> str:
        """The embedding model name for the currently selected provider."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.ollama_embedding_model

    @property
    def llm_model(self) -> str:
        """The chat model name for the currently selected provider."""
        if self.llm_provider == "openai":
            return self.openai_llm_model
        return self.ollama_llm_model

    def model_post_init(self, __context: object) -> None:
        if self.embedding_dim:
            return
        resolved = KNOWN_EMBEDDING_DIMS.get((self.embedding_provider, self.embedding_model))
        if resolved is None:
            raise ValueError(
                f"Unknown embedding dimension for provider '{self.embedding_provider}' "
                f"and model '{self.embedding_model}'. Set EMBEDDING_DIM in .env, or run "
                f"`uv run python scripts/check_embedder.py` to discover it."
            )
        self.embedding_dim = resolved


settings = Settings()
