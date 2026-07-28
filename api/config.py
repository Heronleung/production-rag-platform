"""Centralised configuration, loaded from environment variables or a .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_chunks"
    milvus_metric_type: str = "COSINE"
    milvus_hnsw_m: int = 16
    milvus_hnsw_ef_construction: int = 200
    milvus_hnsw_ef_search: int = 64

    # Chroma (migration source only)
    chroma_path: str = "./.chroma"
    chroma_collection: str = "rag_chunks"


settings = Settings()
