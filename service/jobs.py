"""
In-memory job tracker for asynchronous ingestion.

Preprocessing a PDF with Florence-2 takes several minutes — too long for
an HTTP request to wait on. So /ingest kicks off a background task and
returns a job_id immediately. The big project polls /ingest/{job_id}
until the job is done.

This is a deliberately simple implementation: a dict in memory. If the
RAG service restarts, all in-flight jobs are lost (but the user's data
is already partially written to disk). For a graduation-project-grade
service this is fine. A production deployment would use Celery + Redis
or similar for persistent job state.
"""

import threading
import uuid
from typing import Dict, Any, Optional


_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def create_job() -> str:
    """Create a new job entry in the 'queued' state. Return its id."""
    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "",
            "summary": None,
        }
    return job_id


def set_running(job_id: str) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "running"


def set_done(job_id: str, summary: Dict[str, Any]) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["summary"] = summary
            _JOBS[job_id]["message"] = "Ingestion complete."


def set_failed(job_id: str, error: str) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["message"] = error


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _JOBS.get(job_id)
