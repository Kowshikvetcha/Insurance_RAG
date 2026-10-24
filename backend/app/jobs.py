"""Background ingestion jobs. Docling is CPU-bound, so jobs run in a small thread pool and
publish their progress to a Job object that the SSE endpoint polls."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.config import Settings
from app.ingestion.pipeline import ingest
from app.store import SessionStore

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    message: str = "Waiting to start"
    percent: int = 0
    session_id: str | None = None
    error: str | None = None
    eta_seconds: float | None = None  # rough estimate of time left, recomputed at each progress update
    version: int = 0
    created: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "filename", "status", "stage", "message", "percent", "session_id", "error", "eta_seconds", "version")}


class JobManager:
    def __init__(self, store: SessionStore, settings: Settings, workers: int = 2):
        self.store = store
        self.settings = settings
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, data: bytes, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename)
        with self._lock:
            self.jobs[job.id] = job
        self.pool.submit(self._run, job, data)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _update(self, job: Job, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)
            job.version += 1

    def _run(self, job: Job, data: bytes) -> None:
        self._update(job, status="running", stage="parsing", message="Starting", percent=1)
        try:
            session = ingest(data, job.filename, self.settings,
                             lambda stage, msg, pct, eta=None: self._update(
                                 job, stage=stage, message=msg, percent=pct,
                                 eta_seconds=None if eta is None else round(max(eta, 0.0), 1)))
            self.store.add(session)
            self._update(job, status="done", stage="done", message="Ready", percent=100, session_id=session.id, eta_seconds=0.0)
        except ValueError as exc:  # user-facing problems: too many pages, no text, ...
            self._update(job, status="error", error=str(exc), message="Failed")
        except Exception as exc:  # noqa: BLE001
            # Never log document contents; the exception type and message are enough to debug.
            log.exception("Ingestion failed for job %s", job.id)
            self._update(job, status="error", error=f"Ingestion failed: {type(exc).__name__}: {exc}"[:500], message="Failed")

    def prune(self, max_age_s: int = 3600) -> None:
        cutoff = time.time() - max_age_s
        with self._lock:
            for jid in [j.id for j in self.jobs.values() if j.created < cutoff and j.status in {"done", "error"}]:
                del self.jobs[jid]
