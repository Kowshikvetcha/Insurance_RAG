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

    # Observability master switch: tracing to Langfuse + live answer scoring. Off = neither runs.
    # Tracing alone costs no LLM tokens; the live judge (EVAL_ENABLED) does.
    observability_enabled: bool = True

    # Self-hosted Langfuse (infra/langfuse). Tracing is also off when the keys are empty.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Live answer evaluation (LLM-as-judge). Blank provider = the chat LLM's provider; blank model =
    # that provider's cheaper model from CHEAP_JUDGE_MODELS (else the chat model).
    eval_enabled: bool = True
    eval_sample_rate: float = 1.0
    judge_provider: str = ""
    judge_model: str = ""

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


# Default judge per provider: a smaller, cheaper model from the same vendor, so the judge works with
# whatever API key is already configured for chat.
CHEAP_JUDGE_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
    "google_genai": "gemini-2.5-flash",
    "groq": "llama-3.1-8b-instant",
    "mistralai": "mistral-small-latest",
}


def judge_target(settings: "Settings") -> tuple[str, str]:
    """(provider, model) the judge will use after applying the defaults above."""
    provider = settings.judge_provider or settings.llm_provider
    model = settings.judge_model or CHEAP_JUDGE_MODELS.get(provider) or settings.llm_model
    return provider, model


@lru_cache
def get_settings() -> Settings:
    # Provider SDKs (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...) read keys from the process environment,
    # so export .env there too. Real environment variables take precedence over the file.
    load_dotenv(ENV_FILE, override=False)
    return Settings()
