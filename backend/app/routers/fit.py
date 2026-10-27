import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.deps import AppState, get_state
from app.fit_score import fit_score_stream
from app.schemas import FitScoreRequest

router = APIRouter(tags=["fit-score"])
log = logging.getLogger(__name__)


@router.post("/fit-score")
async def fit_score(req: FitScoreRequest, state: AppState = Depends(get_state)):
    session = state.store.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired. Please upload the document again.")

    async def stream():
        try:
            async for kind, payload in fit_score_stream(session, req.profile, state.llm, state.reranker, state.settings):
                yield {"event": kind, "data": json.dumps(payload)}
        except Exception as exc:  # noqa: BLE001
            log.exception("Fit score failed")
            yield {"event": "error", "data": json.dumps(f"{type(exc).__name__}: {exc}"[:400])}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(stream())
