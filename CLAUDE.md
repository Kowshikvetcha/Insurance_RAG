# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Insurance Policy Analyser: a RAG chatbot. A user uploads a health-insurance policy (PDF/DOCX). The backend ingests it on the fly into a per-session, in-memory hybrid index, and answers questions grounded only in that document, with page/table/figure citations. **The ingestion engine is the core of the project**: policies hold their important facts in tables, footnotes, charts and definitions. See `docs/ARCHITECTURE.md` for the full flow and a file-by-file map.

- `backend/`: FastAPI + LangChain 1.x + Docling (Python 3.12, venv at `backend/.venv`)
- `frontend/`: React 19 + TypeScript + Vite (dev server proxies `/api` → `:8000`)

## Commands

All backend commands run from `backend/` with the venv's Python. The venv uses `--system-site-packages`, so call tools as `.venv/Scripts/python -m <tool>` to avoid picking up global copies.

```bash
# backend
.venv/Scripts/python -m uvicorn app.main:app --port 8000          # API on :8000 (/, /health, /docs)
.venv/Scripts/python -m pytest                                     # all 26 tests, ~60 s (runs real Docling models)
.venv/Scripts/python -m pytest -m "not slow"                       # fast unit tests, ~2 s
.venv/Scripts/python tests/fixtures/make_sample_policy.py          # regenerate the sample policy PDF

# frontend (from frontend/)
npm run dev        # UI on http://localhost:5173
npm run build      # tsc -b && vite build (use this as the type check)

# observability (from infra/langfuse/ and backend/)
docker compose up -d                                               # self-hosted Langfuse on http://localhost:3000
.venv/Scripts/python -m evals.run_evals [--limit N] [--no-ragas]   # offline eval suite -> evals/reports/ + Langfuse dataset run
```

`.claude/launch.json` defines `backend` and `frontend` launch configs for previewing.

## Configuration

`backend/.env` (template: `backend/.env.example`) is laid out in three steps: chat LLM, API keys, embeddings. `app/config.py` loads it from an absolute path and also exports it to `os.environ`, because provider SDKs read keys from there.

- `LLM_PROVIDER` / `EMBED_PROVIDER` = `fake` runs everything offline (no keys). Tests always use `offline_settings()` from `tests/conftest.py`.
- Anthropic has no embeddings, so a Claude setup also needs `openai`, `voyage` or `cohere` for embeddings.
- **Never print, log or commit `backend/.env`.** It holds the user's API keys and is gitignored. When inspecting it, mask `*_API_KEY` values.

## Architecture in one screen

```text
upload → routers/documents.py → jobs.JobManager (thread pool) → ingestion/pipeline.ingest():
  docling_parser (PyMuPDF triage, Docling layout + TableFormer + RapidOCR, clipped-cell recovery)
  → tables (absorb footnotes, cross-page merge, normalise: flatten headers, group rows, footnote links)
  → chunking._Builder (heading stack → text/table parents + search units, glossary)
  → figures (logo filter, vision LLM) → enrich (table summaries, optional LLM tags)
  → store.HybridIndex (embeddings matrix + BM25) → SessionStore (in memory, TTL)
chat → routers/chat.py → rag.answer_stream: rewrite → hybrid search (RRF + category boost → parents)
  → rerank → build_context (+ glossary) → LLM stream → SSE: query, token…, citations, done
```

## Conventions and invariants

- **Parser-neutral seam.** Only `ingestion/docling_parser.py` may import Docling. Everything downstream uses the types in `ingestion/models.py`. A new parser = a class that satisfies `Parser` in `parser_base.py`, registered in `get_parser()`.
- **Parents vs. units.** Units (prose chunks, table-row sentences, table/figure summaries) are embedded and searched. The LLM always receives the unit's *parent* (section window, whole table / row group, figure description). Keep `unit.metadata["parent_id"]` valid whenever you change chunking.
- **Tables stay whole and separate from prose.** Every unit starts with a contextual header `[file › section path] (Table: title)`.
- **Grounding rules live in `prompts.py`.** Answer only from the excerpts, cite `[Source N]`, name plan/column and row for table values, surface waiting periods, exclusions, sub-limits, co-pays and footnotes, say "The policy document does not specify this." when the answer is absent, and give no purchase advice. Preserve these when editing prompts.
- **Privacy.** Documents stay in memory only. Never log document text or chunk contents (log exception types and ids only). Don't add disk persistence without the user agreeing to it.
- **The SSE contract** between `rag.answer_stream`/`routers/chat.py` and `frontend/src/api.ts::streamChat` is the event names `query`, `token`, `citations`, `error`, `done`. Change both sides together. The same applies to the report and citation dict shapes in `pipeline.build_report` / `rag.build_context` and their TypeScript types in `api.ts`.
- **Observability.** Only `app/observability.py` imports Langfuse; everything else uses its `Span` wrapper, which is a no-op when `LANGFUSE_*` keys are empty (tests force them empty in `offline_settings()`). Pass `span.callbacks()` in the LangChain config of every new LLM call so it shows up in the trace. Live grading lives in `app/evaluation.py` and runs after streaming as a background task, so it must never raise or delay the answer. Never put document text in the Python logger; traces may contain it because Langfuse is local-only.
- **Judge model** defaults to the chat provider's cheaper model (`config.CHEAP_JUDGE_MODELS`); `judge_target()` resolves it. RAGAS 0.4.3 needs `evals/ragas_compat.install()` before import (LangChain 1.x removed a module it imports).
- **Suggested questions** live only in `frontend/src/faq.ts`. Keep them generic, not tied to one insurer or plan, so they make sense for any uploaded policy.
- **Style.** Python: type hints, dataclasses, short docstrings that explain *why*, a moderate comment density. TypeScript: function components, no UI library, styles in `index.css` using the CSS-variable theme tokens (light and dark).

## Testing notes

- `tests/fixtures/sample_policy.pdf` is fictional and deliberately covers the hard cases: a 2-page benefit table with a 2-row header and spanning/group rows, `*`/`#` footnotes, an overflowing cell ("(caesarean)#"), a premium chart, and a scanned page. When you change ingestion, run the slow tests. They assert concrete cell placements.
- With fake embeddings the dense leg is random, so ranking assertions must rely on BM25 or on parent-level results.
- `app.state.app_state` can be injected before `TestClient(app)` starts; `lifespan` only builds state when none is present.

## Known gaps

- Real LLM/vision/embedding providers have not been exercised end to end yet. Everything has been verified in offline mode.
- `pymupdf4llm`, `llamaparse` and `azure_di` parsers are stubs.
- There is no persistence, auth or evaluation suite yet (see the roadmap in `README.md`).
