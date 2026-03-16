from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    base_url: str = "https://api.openai.com/v1"
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class VoiceConfig(BaseSettings):
    provider: str = "elevenlabs"
    model_config = SettingsConfigDict(env_prefix="VOICE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class EmbeddingConfig(BaseSettings):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    base_url: str = "https://api.openai.com/v1"
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class VectorStoreConfig(BaseSettings):
    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    collection_name: str = "knowledge_base"
    in_memory: bool = False
    local_path: str = ".qdrant"
    retrieval_limit: int = 5
    score_threshold: float = 0.35
    dense_prefetch_limit: int = 20
    sparse_prefetch_limit: int = 20
    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class IngestionConfig(BaseSettings):
    chunk_size: int = 800
    chunk_overlap: int = 120
    bm25_model: str = "Qdrant/bm25"
    model_config = SettingsConfigDict(env_prefix="INGESTION_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class ServerConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    enable_debug_retrieval: bool = False
    model_config = SettingsConfigDict(env_prefix="SERVER_", env_file=".env", env_file_encoding="utf-8", extra="ignore")


class VoiceRagConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> VoiceRagConfig:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_env(cls) -> VoiceRagConfig:
        return cls()
