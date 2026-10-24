"""Factories for the pluggable model providers (chat LLM, vision LLM, embeddings, reranker).

Everything is driven by Settings so switching provider is a .env change, not a code change.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings


def _make_chat(provider: str, model: str, temperature: float) -> BaseChatModel:
    if provider == "fake":
        from app.fakes import EchoChatModel

        return EchoChatModel()
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, model_provider=provider, temperature=temperature)


def get_llm(settings: Settings) -> BaseChatModel:
    return _make_chat(settings.llm_provider, settings.llm_model, settings.llm_temperature)


def get_vision_llm(settings: Settings) -> BaseChatModel:
    provider = settings.vision_provider or settings.llm_provider
    model = settings.vision_model or settings.llm_model
    return _make_chat(provider, model, 0.0)


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embed_provider
    if provider == "fake":
        return DeterministicFakeEmbedding(size=256)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embed_model, chunk_size=settings.embed_batch_size)
    if provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings  # pip install langchain-voyageai

        return VoyageAIEmbeddings(model=settings.embed_model, batch_size=settings.embed_batch_size)
    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings  # pip install langchain-cohere

        return CohereEmbeddings(model=settings.embed_model)
    raise ValueError(f"Unknown EMBED_PROVIDER: {provider}")


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]: ...


class NoopReranker:
    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        return docs[:top_n]


class CohereReranker:
    def __init__(self, model: str):
        from langchain_cohere import CohereRerank  # pip install langchain-cohere

        self._rerank = CohereRerank(model=model)

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        self._rerank.top_n = top_n
        return list(self._rerank.compress_documents(docs, query))


def get_reranker(settings: Settings) -> Reranker:
    if settings.rerank_provider == "cohere":
        return CohereReranker(settings.rerank_model)
    return NoopReranker()
