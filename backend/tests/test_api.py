"""API tests with offline providers. The upload test runs real Docling, so it's marked slow."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.deps import AppState
from app.jobs import JobManager
from app.main import app
from app.providers import NoopReranker, get_llm
from app.store import SessionStore
from tests.conftest import offline_settings


@pytest.fixture(scope="module")
def client():
    settings = offline_settings()
    store = SessionStore(settings.session_ttl_minutes)
    app.state.app_state = AppState(settings, store, JobManager(store, settings), get_llm(settings), NoopReranker())
    with TestClient(app) as c:
        yield c


def _sse_events(text: str):
    events = []
    for raw in text.replace("\r\n", "\n").split("\n\n"):
        ev, data = None, []
        for line in raw.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if ev:
            events.append((ev, json.loads("\n".join(data)) if data else None))
    return events


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_bad_uploads(client):
    assert client.post("/documents", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400


def test_chat_unknown_session_is_404(client):
    r = client.post("/chat", json={"session_id": "nope", "question": "hi", "history": []})
    assert r.status_code == 404


@pytest.mark.slow
def test_upload_ingest_report_and_chat(client, sample_pdf):
    r = client.post("/documents", files={"file": ("sample_policy.pdf", sample_pdf, "application/pdf")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    deadline = time.time() + 600
    while True:
        job = client.get(f"/documents/jobs/{job_id}").json()
        if job["status"] in {"done", "error"} or time.time() > deadline:
            break
        time.sleep(1)
    assert job["status"] == "done", job
    sid = job["session_id"]

    report = client.get(f"/documents/{sid}/report").json()
    assert report["pages"] == 4 and len(report["tables"]) == 1

    fig = next(f for f in report["figures"] if f["has_image"])
    img = client.get(f"/documents/{sid}/figures/{fig['id']}.png")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    r = client.post("/chat", json={"session_id": sid, "question": "What is the room rent limit?", "history": []})
    events = _sse_events(r.text)
    kinds = [e for e, _ in events]
    assert "token" in kinds and "citations" in kinds and kinds[-1] == "done"

    assert client.delete(f"/documents/{sid}").json() == {"deleted": True}
    assert client.get(f"/documents/{sid}/report").status_code == 404
