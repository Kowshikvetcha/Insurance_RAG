"""Application settings, loaded from environment / backend/.env."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # Chat LLM (any provider supported by langchain's init_chat_model, or "fake" for offline dev)
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    llm_temperature: float = 0.0

    # Vision LLM for figures/charts. Empty = reuse the chat LLM settings.
    vision_provider: str = ""
    vision_model: str = ""

    # Hosted embeddings: openai | voyage | cohere | fake
    embed_provider: str = "openai"
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64

    # Reranker: none | cohere
    rerank_provider: str = "none"
    rerank_model: str = "rerank-v3.5"

    # Ingestion
    parser: str = "docling"  # docling | pymupdf4llm (stub) | llamaparse (stub) | azure_di (stub)
    enable_ocr: bool = True
    enable_vision: bool = True
    enable_table_summaries: bool = True
    enable_llm_tagging: bool = False
    max_upload_mb: int = 25
    max_pages: int = 400
    table_group_rows: int = 40  # max body rows per table parent chunk
    llm_max_concurrency: int = 4

    # Retrieval
    retrieve_k: int = 20
    final_k: int = 6
    history_turns: int = 6

    # Sessions
    session_ttl_minutes: int = 120

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


@lru_cache
def get_settings() -> Settings:
    # Provider SDKs (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) read keys from the process environment,
    # so export .env there too. Real environment variables take precedence over the file.
    load_dotenv(ENV_FILE, override=False)
    return Settings()
