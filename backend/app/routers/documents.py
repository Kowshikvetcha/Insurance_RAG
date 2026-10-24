import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse

from app.deps import AppState, get_state
from app.ingestion.parser_base import SUPPORTED_EXTENSIONS
from app.schemas import UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse)
async def upload(file: UploadFile, state: AppState = Depends(get_state)):
    filename = Path(file.filename or "document.pdf").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type. Upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    limit = state.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {state.settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    if filename.lower().endswith(".pdf") and not data.startswith(b"%PDF"):
        raise HTTPException(400, "File does not look like a PDF")
    job = state.jobs.submit(data, filename)
    return UploadResponse(job_id=job.id, filename=filename)


@router.get("/jobs/{job_id}")
def job_status(job_id: str, state: AppState = Depends(get_state)):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, state: AppState = Depends(get_state)):
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")

    async def stream():
        seen = -1
        while True:
            if await request.is_disconnected():
                return
            if job.version != seen:
                seen = job.version
                yield {"event": "progress", "data": json.dumps(job.snapshot())}
                if job.status in {"done", "error"}:
                    return
            await asyncio.sleep(0.25)

    return EventSourceResponse(stream())


@router.get("/{session_id}/report")
def report(session_id: str, state: AppState = Depends(get_state)):
    session = state.store.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired")
    return session.report


@router.get("/{session_id}/figures/{figure_id}.png")
def figure(session_id: str, figure_id: str, state: AppState = Depends(get_state)):
    session = state.store.get(session_id)
    png = session.figures_png.get(figure_id) if session else None
    if not png:
        raise HTTPException(404, "Figure not found")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/{session_id}")
def delete(session_id: str, state: AppState = Depends(get_state)):
    return {"deleted": state.store.delete(session_id)}
