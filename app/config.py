"""Central configuration. Every setting can be overridden via environment variable or .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "doc-intel-agent"

    # If set, every /v1 request must send header `X-API-Key: <value>`.
    api_key: str | None = None

    # --- Embeddings -------------------------------------------------------
    # "hash"   -> offline, deterministic, no API key (dev/tests only, lexical not semantic)
    # "openai" -> any OpenAI-compatible endpoint (OpenAI, LiteLLM gateway, Azure, etc.)
    embedding_provider: Literal["hash", "openai"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    hash_embedding_dim: int = 512

    # --- LLM --------------------------------------------------------------
    # "extractive" -> offline, returns best-matching sentences (dev/tests only)
    # "openai"     -> OpenAI-compatible chat completions
    llm_provider: Literal["extractive", "openai"] = "extractive"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 30.0

    openai_api_key: str | None = None
    openai_base_url: str | None = None  # point this at your LiteLLM gateway if you have one

    # --- Chunking / retrieval --------------------------------------------
    chunk_size: int = Field(800, ge=100, description="max characters per chunk")
    chunk_overlap: int = Field(100, ge=0, description="characters of overlap between chunks")
    top_k: int = Field(4, ge=1, le=20)
    # Drop retrieved chunks scoring below this cosine similarity. Tune it with evals (Week 2).
    min_score: float = Field(0.0, ge=0.0, le=1.0)

    # --- Storage ----------------------------------------------------------
    data_dir: Path = Path("data/sample_docs")
    index_dir: Path = Path("data/index")
    auto_ingest: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if "openai" in (self.embedding_provider, self.llm_provider) and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using the 'openai' provider")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
