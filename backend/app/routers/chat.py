import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.deps import AppState, get_state
from app.rag import Turn, answer_stream
from app.schemas import ChatRequest

router = APIRouter(tags=["chat"])
log = logging.getLogger(__name__)


@router.post("/chat")
async def chat(req: ChatRequest, state: AppState = Depends(get_state)):
    session = state.store.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired. Please upload the document again.")
    history = [Turn(t.role, t.content) for t in req.history]

    async def stream():
        try:
            async for kind, payload in answer_stream(session, req.question, history, state.llm, state.reranker, state.settings, state.judge):
                yield {"event": kind, "data": json.dumps(payload)}
        except Exception as exc:  # noqa: BLE001
            log.exception("Chat failed")
            yield {"event": "error", "data": json.dumps(f"{type(exc).__name__}: {exc}"[:400])}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(stream())
