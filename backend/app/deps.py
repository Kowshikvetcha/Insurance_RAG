"""Shared application state, exposed as FastAPI dependencies (overridable in tests)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from langchain_core.language_models import BaseChatModel

from app.config import Settings
from app.jobs import JobManager
from app.providers import Reranker
from app.store import SessionStore


@dataclass
class AppState:
    settings: Settings
    store: SessionStore
    jobs: JobManager
    llm: BaseChatModel
    reranker: Reranker


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
