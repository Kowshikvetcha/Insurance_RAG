"""Tracing and scores, backed by a self-hosted Langfuse (see infra/langfuse).

This is the only module that imports Langfuse. When no keys are configured every call is a
no-op, so tests, offline mode and installs without Langfuse behave exactly as before.

Spans are explicit objects rather than context managers: the chat path is an async generator
that yields between steps, and ingestion runs in worker threads, so ambient OpenTelemetry
context is not reliable there. LangChain calls are attached to a span by passing
`span.callbacks()` in the runnable config.

Traces contain policy text. They stay on this machine (the Langfuse ports are bound to
localhost); nothing here writes document text to the Python logger.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)

_client = None
_public_key = ""
_lock = threading.Lock()


def configure(settings: Settings) -> bool:
    """Create the Langfuse client once. Returns whether tracing is enabled."""
    global _client, _public_key
    with _lock:
        if _client is not None:
            return True
        if not settings.observability_enabled:
            log.info("Observability disabled (OBSERVABILITY_ENABLED=false)")
            return False
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return False
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_host,
            )
            _public_key = settings.langfuse_public_key
            log.info("Langfuse tracing enabled (%s)", settings.langfuse_host)
            return True
        except Exception as exc:  # noqa: BLE001 - observability must never break the app
            log.warning("Langfuse disabled: %s", type(exc).__name__)
            _client = None
            return False


def enabled() -> bool:
    return _client is not None


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception as exc:  # noqa: BLE001
            log.warning("Langfuse flush failed: %s", type(exc).__name__)


def short_hash(value: str) -> str:
    """Stable, non-reversible id for grouping traces (e.g. by session) without exposing the raw id."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


class Span:
    """A trace or child observation. All methods are safe no-ops when tracing is disabled."""

    def __init__(self, obs: Any = None):
        self._obs = obs

    @property
    def trace_id(self) -> str | None:
        return getattr(self._obs, "trace_id", None)

    def child(self, name: str, *, as_type: str = "span", input: Any = None, metadata: Any = None) -> "Span":
        if self._obs is None:
            return Span()
        try:
            return Span(self._obs.start_observation(name=name, as_type=as_type, input=input, metadata=metadata))
        except Exception as exc:  # noqa: BLE001
            log.debug("child span failed: %s", type(exc).__name__)
            return Span()

    def update(self, **fields: Any) -> None:
        if self._obs is not None:
            try:
                self._obs.update(**fields)
            except Exception as exc:  # noqa: BLE001
                log.debug("span update failed: %s", type(exc).__name__)

    def end(self, **fields: Any) -> None:
        if self._obs is not None:
            try:
                if fields:
                    self._obs.update(**fields)
                self._obs.end()
            except Exception as exc:  # noqa: BLE001
                log.debug("span end failed: %s", type(exc).__name__)

    def callbacks(self) -> list:
        """LangChain callbacks that record LLM calls as generations under this span."""
        if self._obs is None:
            return []
        try:
            from langfuse.langchain import CallbackHandler

            return [CallbackHandler(public_key=_public_key, trace_context={"trace_id": self._obs.trace_id, "parent_span_id": self._obs.id})]
        except Exception as exc:  # noqa: BLE001
            log.debug("callback handler failed: %s", type(exc).__name__)
            return []

    def score(self, name: str, value: float | str | bool, *, comment: str | None = None, data_type: str | None = None) -> None:
        """Attach a trace-level score (visible in Langfuse's scores views and filters)."""
        if _client is None or self.trace_id is None:
            return
        if isinstance(value, bool):
            value, data_type = (1.0 if value else 0.0), data_type or "BOOLEAN"
        try:
            _client.create_score(name=name, value=value, trace_id=self.trace_id, comment=comment, data_type=data_type)
        except Exception as exc:  # noqa: BLE001
            log.debug("score failed: %s", type(exc).__name__)


def start_trace(name: str, *, input: Any = None, metadata: dict | None = None, session_id: str | None = None,
                tags: list[str] | None = None) -> Span:
    """Start a new trace; its root observation's input/output become the trace's input/output."""
    if _client is None:
        return Span()
    try:
        from langfuse import propagate_attributes

        with propagate_attributes(trace_name=name, session_id=session_id, tags=tags, metadata=_str_values(metadata)):
            return Span(_client.start_observation(name=name, as_type="span", input=input, metadata=metadata))
    except Exception as exc:  # noqa: BLE001
        log.warning("trace start failed: %s", type(exc).__name__)
        return Span()


def _str_values(metadata: dict | None) -> dict | None:
    # Propagated trace attributes must be strings.
    return {k: str(v) for k, v in metadata.items()} if metadata else None
