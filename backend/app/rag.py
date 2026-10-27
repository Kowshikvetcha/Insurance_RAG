"""Answering: query rewrite → hybrid retrieval → rerank → parent expansion + glossary →
grounded, streamed answer with citations.

Each question is one Langfuse trace (when configured) with rewrite / hybrid_search / rerank /
answer observations; after the answer has streamed, a background task grades it and attaches
the scores (see app/evaluation.py)."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import observability as obs
from app.config import Settings
from app.evaluation import evaluate_answer, record_scores
from app.ingestion.enrich import find_terms, query_categories
from app.prompts import ANSWER_TEMPLATE, REWRITE_PROMPT, SYSTEM_PROMPT
from app.providers import Reranker
from app.store import Hit, Session

MAX_CONTEXT_CHARS = 24_000

log = logging.getLogger(__name__)
_background: set[asyncio.Task] = set()  # keep references so evaluation tasks aren't garbage-collected


@dataclass
class Turn:
    role: str  # user | assistant
    content: str


def _history_text(history: list[Turn], turns: int) -> str:
    recent = history[-turns:]
    return "\n".join(f"{t.role.capitalize()}: {t.content[:1200]}" for t in recent)


async def rewrite_query(llm: BaseChatModel, question: str, history: list[Turn], settings: Settings,
                        callbacks: list | None = None) -> str:
    if not history:
        return question
    prompt = REWRITE_PROMPT.format(history=_history_text(history, settings.history_turns), question=question)
    try:
        reply = await llm.ainvoke([HumanMessage(content=prompt)], config={"callbacks": callbacks or [], "run_name": "rewrite"})
        text = str(reply.content).strip().strip('"')
        return text or question
    except Exception:  # noqa: BLE001 - fall back to the raw question
        return question


def hit_summary(n: int, h: Hit) -> dict:
    """Compact, trace-friendly view of one retrieved parent."""
    p = h.parent
    return {
        "source": n,
        "parent_id": p.id,
        "type": p.type,
        "pages": p.pages,
        "title": p.title,
        "section": " › ".join(p.section_path),
        "categories": p.categories,
        "score": round(h.score, 5),
        "dense_rank": h.dense_rank,
        "dense_sim": None if h.dense_sim is None else round(h.dense_sim, 4),
        "bm25_rank": h.bm25_rank,
        "matched": h.best_unit.page_content.split("\n", 1)[-1][:200],
    }


def retrieve(session: Session, query: str, reranker: Reranker, settings: Settings, trace: obs.Span | None = None) -> list[Hit]:
    trace = trace or obs.Span()
    categories = query_categories(query)
    span = trace.child("hybrid_search", as_type="retriever", input=query,
                       metadata={"retrieve_k": settings.retrieve_k, "boost_categories": sorted(categories)})
    hits = session.index.search(query, settings.retrieve_k, boost_categories=categories)
    candidates = hits[: max(settings.final_k * 2, 10)]
    span.end(output=[hit_summary(i, h) for i, h in enumerate(candidates, start=1)])

    span = trace.child("rerank", input={"candidates": len(candidates)},
                       metadata={"reranker": type(reranker).__name__, "final_k": settings.final_k})
    docs = [Document(page_content=h.parent.text, metadata={"pid": h.parent.id}) for h in candidates]
    ranked = reranker.rerank(query, docs, settings.final_k)
    by_id = {h.parent.id: h for h in candidates}
    result = [by_id[d.metadata["pid"]] for d in ranked if d.metadata["pid"] in by_id]
    span.end(output=[h.parent.id for h in result])
    return result


def _pages_label(pages: list[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p. {pages[0]}"
    return f"pp. {pages[0]}–{pages[-1]}"


def source_block(n: int, hit: Hit, text: str) -> str:
    """One excerpt as the LLM sees it. `text` may be the parent text truncated to the context cap."""
    p = hit.parent
    section = " › ".join(p.section_path)
    return f'<source id="{n}" type="{p.type}" pages="{_pages_label(p.pages)}" section="{section}">\n{text}\n</source>'


def citation(session: Session, n: int, hit: Hit) -> dict:
    """The citation dict the UI shows for source n (mirrored by `Citation` in frontend/src/api.ts)."""
    p = hit.parent
    snippet = hit.best_unit.page_content.split("\n", 1)[-1]
    return {
        "source": n,
        "type": p.type,
        "pages": p.pages,
        "page_label": _pages_label(p.pages),
        "section": " › ".join(p.section_path),
        "title": p.title,
        "snippet": snippet[:400],
        "content": p.text if p.type in {"table", "figure"} else p.text[:1500],
        "figure_id": p.ref if p.type == "figure" and p.ref in session.figures_png else None,
    }


def glossary_block(session: Session, text: str) -> str:
    terms = find_terms(text, session.glossary)
    return "\n".join(f"- {d.definition} (p. {d.page})" if d.page else f"- {d.definition}" for d in terms) or "(none)"


def build_context(session: Session, question: str, hits: list[Hit]) -> tuple[str, str, list[dict]]:
    excerpts: list[str] = []
    citations: list[dict] = []
    used = 0
    for n, hit in enumerate(hits, start=1):
        text = hit.parent.text
        if used + len(text) > MAX_CONTEXT_CHARS:
            text = text[: max(0, MAX_CONTEXT_CHARS - used)]
            if not text:
                break
        used += len(text)
        excerpts.append(source_block(n, hit, text))
        citations.append(citation(session, n, hit))
    definitions = glossary_block(session, question + "\n" + "\n".join(excerpts))
    return "\n\n".join(excerpts), definitions, citations


async def answer_stream(session: Session, question: str, history: list[Turn], llm: BaseChatModel,
                        reranker: Reranker, settings: Settings,
                        judge: BaseChatModel | None = None) -> AsyncIterator[tuple[str, object]]:
    """Yields ("query", str), then ("token", str)*, then ("citations", list)."""
    trace = obs.start_trace(
        "chat",
        input={"question": question, "history_turns": len(history)},
        metadata={"filename": session.filename,
                  "llm": f"{settings.llm_provider}:{settings.llm_model}",
                  "embeddings": f"{settings.embed_provider}:{settings.embed_model}"},
        session_id=obs.short_hash(session.id),
        tags=["chat"],
    )
    parts: list[str] = []
    hits: list[Hit] = []
    excerpts = ""
    completed = False
    try:
        span = trace.child("rewrite", input=question)
        query = await rewrite_query(llm, question, history, settings, callbacks=span.callbacks())
        span.end(output=query)
        yield "query", query

        hits = await asyncio.to_thread(retrieve, session, query, reranker, settings, trace)
        excerpts, definitions, citations = build_context(session, query, hits)
        hits = hits[: len(citations)]  # sources dropped by the context cap were never shown to the LLM
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for turn in history[-settings.history_turns:]:
            messages.append(HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content))
        messages.append(HumanMessage(content=ANSWER_TEMPLATE.format(definitions=definitions, excerpts=excerpts or "(no excerpts found)", question=question)))
        async for chunk in llm.astream(messages, config={"callbacks": trace.callbacks(), "run_name": "answer"}):
            text = chunk.text  # str property on langchain-core 1.x message chunks
            if text:
                parts.append(text)
                yield "token", text
        yield "citations", citations
        completed = True
    except Exception as exc:
        trace.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:300])
        raise
    finally:
        answer = "".join(parts)
        trace.end(output={"answer": answer, "sources": [hit_summary(i, h) for i, h in enumerate(hits, start=1)]})
        if completed and obs.enabled():
            # Free heuristic scores are always recorded; the judge (LLM tokens) only when enabled and sampled.
            task = asyncio.create_task(_evaluate(trace, question, excerpts, answer, hits, _judge_for_this_answer(settings, judge)))
            _background.add(task)
            task.add_done_callback(_background.discard)


def _judge_for_this_answer(settings: Settings, judge: BaseChatModel | None) -> BaseChatModel | None:
    if judge is None or not (settings.observability_enabled and settings.eval_enabled):
        return None
    return judge if random.random() < settings.eval_sample_rate else None


async def _evaluate(trace: obs.Span, question: str, excerpts: str, answer: str, hits: list[Hit],
                    judge: BaseChatModel | None) -> None:
    """Grade a finished answer and attach the scores to its trace. Never raises."""
    try:
        span = trace.child("evaluation", as_type="evaluator", input={"question": question, "sources": len(hits)})
        result = await evaluate_answer(question, excerpts, answer, hits, judge, callbacks=span.callbacks())
        span.end(output=result.as_dict(), level="WARNING" if result.judge_error else "DEFAULT")
        record_scores(trace, result)
        # Numbers only: document text never goes to the application log.
        log.info("answer eval: confidence=%s (%s) groundedness=%s context_relevance=%s answer_relevance=%s "
                 "citations_valid=%s not_in_document=%s", result.confidence, result.confidence_label,
                 result.groundedness, result.context_relevance, result.answer_relevance,
                 result.citation_validity, result.not_in_document)
    except Exception:  # noqa: BLE001
        log.exception("answer evaluation failed")
