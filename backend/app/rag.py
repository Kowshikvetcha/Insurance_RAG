"""Answering: query rewrite → hybrid retrieval → rerank → parent expansion + glossary →
grounded, streamed answer with citations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.ingestion.enrich import find_terms, query_categories
from app.prompts import ANSWER_TEMPLATE, REWRITE_PROMPT, SYSTEM_PROMPT
from app.providers import Reranker
from app.store import Hit, Session

MAX_CONTEXT_CHARS = 24_000


@dataclass
class Turn:
    role: str  # user | assistant
    content: str


def _history_text(history: list[Turn], turns: int) -> str:
    recent = history[-turns:]
    return "\n".join(f"{t.role.capitalize()}: {t.content[:1200]}" for t in recent)


async def rewrite_query(llm: BaseChatModel, question: str, history: list[Turn], settings: Settings) -> str:
    if not history:
        return question
    prompt = REWRITE_PROMPT.format(history=_history_text(history, settings.history_turns), question=question)
    try:
        reply = await llm.ainvoke([HumanMessage(content=prompt)])
        text = str(reply.content).strip().strip('"')
        return text or question
    except Exception:  # noqa: BLE001 - fall back to the raw question
        return question


def retrieve(session: Session, query: str, reranker: Reranker, settings: Settings) -> list[Hit]:
    hits = session.index.search(query, settings.retrieve_k, boost_categories=query_categories(query))
    candidates = hits[: max(settings.final_k * 2, 10)]
    docs = [Document(page_content=h.parent.text, metadata={"pid": h.parent.id}) for h in candidates]
    ranked = reranker.rerank(query, docs, settings.final_k)
    by_id = {h.parent.id: h for h in candidates}
    return [by_id[d.metadata["pid"]] for d in ranked if d.metadata["pid"] in by_id]


def _pages_label(pages: list[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p. {pages[0]}"
    return f"pp. {pages[0]}–{pages[-1]}"


def build_context(session: Session, question: str, hits: list[Hit]) -> tuple[str, str, list[dict]]:
    excerpts: list[str] = []
    citations: list[dict] = []
    used = 0
    for n, hit in enumerate(hits, start=1):
        p = hit.parent
        text = p.text
        if used + len(text) > MAX_CONTEXT_CHARS:
            text = text[: max(0, MAX_CONTEXT_CHARS - used)]
            if not text:
                break
        used += len(text)
        section = " › ".join(p.section_path)
        excerpts.append(f'<source id="{n}" type="{p.type}" pages="{_pages_label(p.pages)}" section="{section}">\n{text}\n</source>')
        snippet = hit.best_unit.page_content.split("\n", 1)[-1]
        citations.append({
            "source": n,
            "type": p.type,
            "pages": p.pages,
            "page_label": _pages_label(p.pages),
            "section": section,
            "title": p.title,
            "snippet": snippet[:400],
            "content": p.text if p.type in {"table", "figure"} else p.text[:1500],
            "figure_id": p.ref if p.type == "figure" and p.ref in session.figures_png else None,
        })
    terms = find_terms(question + "\n" + "\n".join(excerpts), session.glossary)
    definitions = "\n".join(f"- {d.definition} (p. {d.page})" if d.page else f"- {d.definition}" for d in terms) or "(none)"
    return "\n\n".join(excerpts), definitions, citations


async def answer_stream(session: Session, question: str, history: list[Turn], llm: BaseChatModel,
                        reranker: Reranker, settings: Settings) -> AsyncIterator[tuple[str, object]]:
    """Yields ("query", str), then ("token", str)*, then ("citations", list)."""
    query = await rewrite_query(llm, question, history, settings)
    yield "query", query
    hits = await asyncio.to_thread(retrieve, session, query, reranker, settings)
    excerpts, definitions, citations = build_context(session, query, hits)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history[-settings.history_turns:]:
        messages.append(HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content))
    messages.append(HumanMessage(content=ANSWER_TEMPLATE.format(definitions=definitions, excerpts=excerpts or "(no excerpts found)", question=question)))
    async for chunk in llm.astream(messages):
        text = chunk.text  # str property on langchain-core 1.x message chunks
        if text:
            yield "token", text
    yield "citations", citations

