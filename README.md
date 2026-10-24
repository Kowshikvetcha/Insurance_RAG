# Insurance Policy Analyser

Upload a health insurance policy (PDF or DOCX) and ask questions about it. Every answer is grounded in
the document and cites the page, table or figure it came from. Retrieval runs over embeddings built
when you upload; there's no pre-built index.

The core of the project is the **ingestion engine**. Insurance documents keep their important facts
in tables (benefit schedules, sub-limits, plan comparisons) and in charts, footnotes and definitions.
Flattened text can't preserve any of that structure.

## Ingestion pipeline

| Stage | What it does | Code |
|---|---|---|
| Parse | Docling layout model (headings, reading order, multi-column), **TableFormer** table structure (merged/spanning cells, multi-row headers), OCR for image-only pages, figure crops. PyMuPDF triage flags scanned pages. **Clipped-text recovery** puts back words that overflow a cell border, which TableFormer would otherwise drop. | `app/ingestion/docling_parser.py` |
| Tables | Merges tables that continue on the next page. Flattens multi-row headers (`Plan Variant › Gold`). Turns full-width group rows into row-label prefixes and forward-fills merged row labels. Links footnote markers (`*`, `#`, `¹`) in cells to their footnote text. | `app/ingestion/tables.py` |
| Figures | Drops logos and icons by size. Describes charts and diagrams with the vision LLM (text, numbers as a table, flow steps). Falls back to caption + OCR text. | `app/ingestion/figures.py` |
| Enrichment | Builds a definitions glossary that is injected at answer time, section category tags (coverage / exclusion / waiting period / sub-limit / co-pay / claims / …), and LLM table summaries. | `app/ingestion/enrich.py` |
| Chunking | Two-level index. **Parents** (full section windows, full tables with the header repeated per row group, figure descriptions) are what the LLM reads. **Units** (prose chunks, one sentence per table row, table summaries) are what gets searched. Every unit carries a `document › section › table` header. | `app/ingestion/chunking.py` |
| Index | Dense embeddings + BM25 with reciprocal-rank fusion, category boosts, and collapse to parents. In memory, per session, with a TTL. | `app/store.py` |

Answering (`app/rag.py`): history-aware query rewrite → hybrid retrieval → optional rerank →
parent expansion + glossary definitions → grounded prompt (`app/prompts.py`) → streamed answer +
citations.

After upload, the UI shows an **ingestion report**: tables (with previews so you can check the
extraction), figures, OCR pages, defined terms and warnings.

## Run it

Prerequisites: Python 3.12 and Node.js 20+ (developed on Node 24). No database or other services are needed.

**1. Backend** (terminal 1):

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env    # then edit it (see Configuration below)
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Port 8000 is the API only. `http://127.0.0.1:8000/` confirms it is running, `/health` is a health check, and
`/docs` is the interactive API docs. Run uvicorn through the venv's Python as shown: a bare `uvicorn` can
resolve to a global install that lacks the project's packages.

**2. Frontend** (terminal 2):

```bash
cd frontend
npm install
npm run dev
```

**3.** Open **<http://localhost:5173>**. The dev server proxies `/api/*` to the backend on :8000.

The first upload loads the Docling models (about 1 minute). After that, a 4-page policy takes about 19 s on CPU,
plus the LLM calls for figures and table summaries.

### Configuration (`backend/.env`)

The file is laid out in three commented steps:

1. **Chat LLM**: `LLM_PROVIDER` / `LLM_MODEL`. Any `init_chat_model` provider works: `anthropic`, `openai`,
   `google_genai`, `groq`, `mistralai`, `ollama`. The table in the file lists the key each one needs.
   `VISION_PROVIDER` / `VISION_MODEL` choose the chart reader; leave them blank to reuse the chat LLM (it must accept images).
2. **API keys**: paste the key for each provider you chose and leave the others empty.
3. **Embeddings**: `EMBED_PROVIDER` = `openai` | `voyage` | `cohere`. Anthropic has no embeddings API, so a Claude
   setup also needs one of these. `RERANK_PROVIDER` = `none` | `cohere`.

- Setting `fake` for both the LLM and embeddings runs fully offline: answers echo the best excerpt, which is useful for
  checking ingestion and the UI without keys.
- Cost switches: `ENABLE_VISION`, `ENABLE_TABLE_SUMMARIES`, `ENABLE_LLM_TAGGING`. Limits: `MAX_UPLOAD_MB`, `MAX_PAGES`,
  `SESSION_TTL_MINUTES`.
- Restart the backend after editing `.env`. Never commit it: it is gitignored because it holds your keys.

## Project layout

```text
backend/app/            FastAPI app: config, providers, store, rag, prompts, routers
backend/app/ingestion/  ingestion engine: parser, tables, figures, enrich, chunking, pipeline
backend/tests/          pytest suite + synthetic policy fixture
frontend/src/           React UI: api client, App, components
docs/ARCHITECTURE.md    request flows, file-by-file reference, design decisions
CLAUDE.md               guidance for Claude Code in this repo
```

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest              # all tests (~1 min; runs the real Docling models)
.venv/Scripts/python -m pytest -m "not slow"   # fast unit tests only
```

`tests/fixtures/make_sample_policy.py` generates a fictional policy that covers the hard cases: a
2-page benefit table with a 2-row header and spanning cells, group rows, footnotes, an overflowing
cell, a chart and a scanned page.

## Privacy

Policy documents can contain personal data (names, policy and member numbers):

- Documents are processed in memory only and are never written to disk.
- Document text is never logged.
- Sessions expire after `SESSION_TTL_MINUTES`, and `DELETE /documents/{id}` removes one immediately.
- Document text and figure images **are sent to the configured embedding and LLM/vision providers**.
  Use `ollama` for the LLM if that's not acceptable.

## Further documentation

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md): what the project does and why, the ingestion and answering pipeline, UI components, tools used at each stage.
- [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md): engineering design: problem statement, requirements, design decisions with alternatives, ingestion and retrieval internals, API contracts, performance and ETA model, libraries and versions, testing, limitations.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how upload and chat requests flow through the code, a file-by-file reference, and the design decisions and gotchas.
- [CLAUDE.md](CLAUDE.md): commands, conventions and invariants for working on the repo with Claude Code.
