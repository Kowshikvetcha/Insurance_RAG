"""Ingestion orchestration: parse → tables → structure → figures → enrichment → embedding.

Reports progress through a callback (stage, message, percent, eta_seconds) and returns a ready
Session plus an ingestion report the UI shows so the user can sanity-check what was extracted.

The ETA is a rough estimate. Parsing dominates on CPU, so it is measured live from the page batches
and carried over between uploads; the remaining stages are estimated from their workload."""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

from app.config import Settings
from app.ingestion import chunking
from app.ingestion.enrich import llm_tag, summarise_tables
from app.ingestion.figures import describe_figures, skip_reason
from app.ingestion.parser_base import get_parser
from app.ingestion.tables import absorb_trailing_footnotes, merge_cross_page_tables
from app.providers import get_embeddings, get_llm, get_vision_llm
from app.store import HybridIndex, Session, new_session_id

log = logging.getLogger(__name__)

Progress = Callable[[str, str, int, "float | None"], None]  # stage, message, percent, eta seconds

STAGES = ["parsing", "tables", "figures", "enrichment", "embedding"]

# Percent of the progress bar where each stage starts; parsing fills 3..60 page by page.
PCT = {"parsing": 3, "tables": 60, "figures": 64, "enrichment": 74, "embedding": 82}

# ETA model (seconds). SEC_PER_PAGE is learned from previous uploads in this process.
_rate = {"sec_per_page": 3.0}
MODEL_LOAD_SEC = 20.0
VISION_SEC = 8.0  # one figure description
SUMMARY_SEC = 5.0  # one table summary
EMBED_BATCH_SEC = 1.0


def _after_parse_estimate(pages: int, settings: Settings) -> float:
    """Rough time for everything after parsing, before we know how many tables/figures there are."""
    llm_on = settings.llm_provider != "fake"
    per_page = 0.15 + (0.6 if llm_on and (settings.enable_vision or settings.enable_table_summaries) else 0.0)
    return 2.0 + per_page * pages


def _llm_batch_sec(n: int, unit_sec: float, settings: Settings) -> float:
    return math.ceil(n / max(1, settings.llm_max_concurrency)) * unit_sec if n else 0.0


def _embed_sec(units: int, settings: Settings) -> float:
    batches = math.ceil(units / max(1, settings.embed_batch_size))
    return math.ceil(batches / 4) * EMBED_BATCH_SEC + 0.5


def _llm_or_none(settings: Settings, enabled: bool, vision: bool = False):
    provider = (settings.vision_provider or settings.llm_provider) if vision else settings.llm_provider
    if not enabled or provider == "fake":
        return None
    return get_vision_llm(settings) if vision else get_llm(settings)


def ingest(data: bytes, filename: str, settings: Settings, progress: Progress) -> Session:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. parse ------------------------------------------------------------------------------
    progress("parsing", "Analysing layout", PCT["parsing"], None)
    parser = get_parser(settings)
    parse_start: list[float] = []  # set when the first page batch starts (after any model loading)

    def on_pages(done: int, total: int, msg: str) -> None:
        now = time.perf_counter()
        if msg.startswith("Loading"):
            eta = MODEL_LOAD_SEC + _rate["sec_per_page"] * total + _after_parse_estimate(total, settings)
        else:
            if not parse_start:
                parse_start.append(now)
            elapsed = now - parse_start[0]
            # Blend the live rate with the learned one; trust the live rate more as pages complete.
            live = elapsed / done if done else _rate["sec_per_page"]
            weight = min(1.0, done / 4)
            rate = weight * live + (1 - weight) * _rate["sec_per_page"]
            eta = rate * (total - done) + _after_parse_estimate(total, settings)
        pct = PCT["parsing"] + int((PCT["tables"] - PCT["parsing"]) * done / max(total, 1))
        progress("parsing", msg, pct, eta)

    doc = parser.parse(data, filename, on_pages)
    timings["parsing"] = time.perf_counter() - t0
    if parse_start and doc.num_pages:
        measured = (time.perf_counter() - parse_start[0]) / doc.num_pages
        _rate["sec_per_page"] = 0.5 * _rate["sec_per_page"] + 0.5 * measured
    raw_tables = sum(1 for b in doc.blocks if b.kind == "table")

    # 2. tables -----------------------------------------------------------------------------
    t = time.perf_counter()
    progress("tables", f"Normalising {raw_tables} table fragment(s)", PCT["tables"], _after_parse_estimate(doc.num_pages, settings))
    doc.blocks = merge_cross_page_tables(absorb_trailing_footnotes(doc.blocks))
    builder = chunking.build_structure(doc, settings.table_group_rows)
    timings["tables"] = time.perf_counter() - t

    # 3. figures ----------------------------------------------------------------------------
    t = time.perf_counter()
    for rec in builder.figures:
        rec.figure.skipped_reason = skip_reason(rec.figure)
    meaningful = [r for r in builder.figures if not r.figure.skipped_reason]
    vision = _llm_or_none(settings, settings.enable_vision, vision=True)
    summariser = _llm_or_none(settings, settings.enable_table_summaries)
    units_guess = len(builder.units) + len(builder.tables) + len(meaningful)
    rest = (_llm_batch_sec(len(builder.tables), SUMMARY_SEC, settings) if summariser else 0.0) + _embed_sec(units_guess, settings)
    figures_sec = _llm_batch_sec(len(meaningful), VISION_SEC, settings) if vision else 0.0
    progress("figures", f"Describing {len(meaningful)} figure(s)" if vision and meaningful else f"Reading {len(meaningful)} figure(s)",
             PCT["figures"], figures_sec + rest)
    described = describe_figures([(r.figure, " › ".join(r.section_path)) for r in builder.figures], vision, settings.llm_max_concurrency)
    builder.finish_figures()
    timings["figures"] = time.perf_counter() - t

    # 4. enrichment -------------------------------------------------------------------------
    t = time.perf_counter()
    progress("enrichment", f"Summarising {len(builder.tables)} table(s) and tagging sections", PCT["enrichment"], rest)
    summarised = summarise_tables(builder.tables, summariser, settings.llm_max_concurrency)
    builder.add_table_summaries()
    tagger = _llm_or_none(settings, settings.enable_llm_tagging)
    if tagger is not None:
        _retag_general(builder, tagger)
    timings["enrichment"] = time.perf_counter() - t

    # 5. embedding --------------------------------------------------------------------------
    t = time.perf_counter()
    chunked = chunking.to_chunked(builder)
    if not chunked.units:
        raise ValueError("No text could be extracted from this document.")
    index = HybridIndex(chunked.units, chunked.parents, get_embeddings(settings))
    progress("embedding", f"Embedding {len(chunked.units)} search units", PCT["embedding"], _embed_sec(len(chunked.units), settings))
    embed_start = time.perf_counter()

    def on_embedded(done: int, total: int) -> None:
        elapsed = time.perf_counter() - embed_start
        eta = elapsed / done * (total - done) if done else None
        progress("embedding", f"Embedded {done} of {total} search units", PCT["embedding"] + int((99 - PCT["embedding"]) * done / total), eta)

    index.embed(settings.embed_batch_size, on_embedded)
    timings["embedding"] = time.perf_counter() - t

    figures_png = {r.id: r.figure.image_png for r in builder.figures if r.figure.image_png and not r.figure.skipped_reason}
    report = build_report(doc, chunked, raw_tables, described, summarised, timings, time.perf_counter() - t0, settings)
    progress("done", "Ready", 100, 0.0)
    return Session(id=new_session_id(), filename=filename, index=index, glossary=chunked.glossary, report=report, figures_png=figures_png)


def _retag_general(builder: chunking._Builder, llm, batch: int = 15) -> None:
    general = [p for p in builder.parents.values() if p.categories == ["general"]]
    for i in range(0, len(general), batch):
        group = general[i : i + batch]
        try:
            tags = llm_tag([p.text for p in group], llm)
        except Exception as exc:  # noqa: BLE001 - tagging is best effort
            log.warning("LLM tagging failed: %s", type(exc).__name__)
            return
        for parent, cats in zip(group, tags):
            parent.categories = cats
    for unit in builder.units:
        unit.metadata["categories"] = builder.parents[unit.metadata["parent_id"]].categories


def build_report(doc, chunked: chunking.ChunkedDocument, raw_tables: int, described: int, summarised: int,
                 timings: dict[str, float], total: float, settings: Settings) -> dict:
    tables = []
    for tbl in chunked.tables:
        tables.append({
            "id": tbl.id,
            "title": tbl.title,
            "pages": tbl.page_label(),
            "rows": len(tbl.rows),
            "cols": len(tbl.columns),
            "merged_from": tbl.merged_from,
            "footnotes": len(tbl.footnotes),
            "preview": tbl.to_markdown(0, 8, with_footnotes=False),
            "summary": tbl.summary,
        })
    figures = [{
        "id": r.id,
        "page": r.page,
        "caption": r.figure.caption,
        "status": "skipped" if r.figure.skipped_reason else (
            "described" if r.figure.described_by_vision else ("caption/OCR text only" if r.figure.description else "no description")),
        "reason": r.figure.skipped_reason,
        "has_image": bool(r.figure.image_png) and not r.figure.skipped_reason,
    } for r in chunked.figures]
    warnings = list(doc.warnings)
    if raw_tables and not chunked.tables:
        warnings.append("Table regions were detected but none had a usable row/column structure; they were indexed as text.")
    if settings.enable_vision and settings.llm_provider != "fake" and any(f["status"] not in {"described", "skipped"} for f in figures):
        warnings.append("Some figures could not be described by the vision model; their captions/OCR text were used instead.")
    unit_types: dict[str, int] = {}
    for u in chunked.units:
        unit_types[u.metadata["type"]] = unit_types.get(u.metadata["type"], 0) + 1
    return {
        "filename": doc.filename,
        "parser": doc.parser,
        "pages": doc.num_pages,
        "ocr_pages": doc.ocr_pages,
        "tables": tables,
        "table_fragments": raw_tables,
        "tables_summarised_by_llm": summarised,
        "figures": figures,
        "figures_described_by_llm": described,
        "definitions": len(chunked.glossary),
        "definition_terms": sorted(d.term for d in chunked.glossary.values())[:200],
        "sections": len({tuple(p.section_path) for p in chunked.parents.values()}),
        "parents": len(chunked.parents),
        "search_units": unit_types,
        "warnings": warnings,
        "timings": {k: round(v, 2) for k, v in timings.items()} | {"total": round(total, 2)},
    }
