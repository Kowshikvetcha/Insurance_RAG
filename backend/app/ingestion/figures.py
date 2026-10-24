"""Figures and charts: drop decorative images, describe the rest with a vision LLM."""

from __future__ import annotations

import base64
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.ingestion.models import Figure

log = logging.getLogger(__name__)

MIN_SIDE_PT = 60  # smaller than this (in PDF points) is an icon/logo/bullet
MIN_AREA_PT = 12_000

VISION_PROMPT = """You are reading a figure from a health insurance policy document.
1. In one or two sentences, say what the figure shows (chart, flow diagram, infographic, table image, ...).
2. Transcribe ALL text in the figure exactly.
3. If it contains numbers or data series (premiums, limits, percentages, ages), reproduce them as a markdown table with clear column headers.
4. If it is a process/flow diagram, list the steps in order.
Do not add information that is not visible in the figure.
{context}"""


def skip_reason(fig: Figure) -> str:
    if fig.width and fig.height:
        if min(fig.width, fig.height) < MIN_SIDE_PT or fig.width * fig.height < MIN_AREA_PT:
            return "decorative (too small: logo/icon)"
    if fig.image_png is None and not fig.caption and not fig.inner_text:
        return "no image or text available"
    return ""


def describe_figures(figures: list[tuple[Figure, str]], llm: BaseChatModel | None, max_concurrency: int) -> int:
    """Fill Figure.description in place. `figures` pairs each figure with its section context.
    Returns how many were described by the vision model."""
    todo = [(f, ctx) for f, ctx in figures if not f.skipped_reason and f.image_png]
    described = 0
    if llm is not None and todo:
        messages = []
        for fig, ctx in todo:
            context = f"Context: this figure appears in section '{ctx}'." if ctx else ""
            if fig.caption:
                context += f" Caption: {fig.caption}"
            messages.append([
                HumanMessage(content=[
                    {"type": "text", "text": VISION_PROMPT.format(context=context)},
                    {"type": "image", "base64": base64.b64encode(fig.image_png).decode(), "mime_type": "image/png"},
                ])
            ])
        results = llm.batch(messages, config={"max_concurrency": max_concurrency}, return_exceptions=True)
        for (fig, _), res in zip(todo, results):
            if isinstance(res, Exception):
                log.warning("Vision description failed: %s", type(res).__name__)
                continue
            text = res.content if isinstance(res.content, str) else str(res.text)
            if text.strip():
                fig.description = text.strip()
                fig.described_by_vision = True
                described += 1
    # Fallback: whatever text we have (caption + text found inside the figure by the parser/OCR)
    for fig, _ in figures:
        if not fig.skipped_reason and not fig.description:
            inner = fig.inner_text
            if fig.caption and inner.startswith(fig.caption):
                inner = inner[len(fig.caption):].strip()
            fig.description = " ".join(p for p in [fig.caption, inner] if p).strip()
    return described
