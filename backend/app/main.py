import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.deps import AppState
from app.ingestion.parser_base import get_parser
from app.jobs import JobManager
from app.providers import get_llm, get_reranker
from app.routers import chat, documents
from app.store import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


def build_state() -> AppState:
    settings = get_settings()
    store = SessionStore(settings.session_ttl_minutes)
    return AppState(
        settings=settings,
        store=store,
        jobs=JobManager(store, settings),
        llm=get_llm(settings),
        reranker=get_reranker(settings),
    )


async def _janitor(state: AppState) -> None:
    while True:
        await asyncio.sleep(60)
        evicted = state.store.evict_expired()
        state.jobs.prune()
        if evicted:
            log.info("Evicted %d expired session(s)", evicted)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "app_state"):
        app.state.app_state = build_state()
    state: AppState = app.state.app_state
    # Load the layout/table/OCR models in the background so the first upload isn't slow.
    threading.Thread(target=lambda: _warmup(state), daemon=True).start()
    janitor = asyncio.create_task(_janitor(state))
    yield
    janitor.cancel()
    state.jobs.pool.shutdown(wait=False, cancel_futures=True)


def _warmup(state: AppState) -> None:
    try:
        get_parser(state.settings).warmup()
        log.info("Parser models ready")
    except Exception:  # noqa: BLE001
        log.exception("Parser warmup failed")


app = FastAPI(title="Insurance Policy Analyser", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Insurance Policy Analyser API is running. Open the chat UI at http://localhost:5173 (cd frontend && npm run dev).",
        "api_docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
