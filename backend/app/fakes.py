"""Offline stand-in chat model (LLM_PROVIDER=fake) so the app runs end to end without API keys.

It does not reason: it echoes back the most relevant retrieved excerpt, which is enough to
exercise ingestion, retrieval, citations and streaming in the UI and in tests.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _fake_reply(messages: list[BaseMessage]) -> str:
    prompt = _text_of(messages[-1])
    if "Follow-up question:" in prompt:  # query-rewrite prompt
        return prompt.rsplit("Follow-up question:", 1)[1].strip()
    excerpt = re.search(r"<source[^>]*>\s*(.*?)\s*</source>", prompt, re.S)
    if excerpt:  # answer prompt
        snippet = re.sub(r"\s+", " ", excerpt.group(1))[:700]
        return f"(Offline mode: no LLM configured.) Most relevant excerpt [Source 1]:\n\n{snippet}"
    if "<table>" in prompt:  # table summary prompt
        return "Table summary unavailable in offline mode."
    return "Offline mode: no LLM configured."


class EchoChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "echo-fake"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_fake_reply(messages)))])

    def _stream(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for token in re.findall(r"\S+\s*", _fake_reply(messages)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
