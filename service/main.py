"""
RAG microservice — FastAPI wrapper around the existing RAG library.

The big project talks to this service over HTTP. The service handles:
  - Ingestion (background task because preprocessing takes minutes)
  - Job status polling
  - Retrieval (synchronous, fast)
  - Collection deletion (when a session is closed)

Auth: a shared API key passed in the X-Internal-Token header. The big
project sets it; the service checks it. Configurable via the env var
RAG_API_TOKEN. If unset, auth is disabled (development mode).

Run:
  uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
  http://localhost:8000/docs
"""

import os
import sys
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.responses import JSONResponse

# Make the RAG library importable when running uvicorn from project root
_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Library imports
from pipeline import ingest_files
from retrieval.retriever import Retriever
from embedding.vector_store import VectorStore

# Service-local imports
from .models import (
    IngestRequest, IngestResponse,
    JobStatusResponse,
    RetrieveRequest, RetrieveResponse, RetrievedChunk,
    DeleteCollectionResponse,
)
from . import jobs


API_TOKEN = os.getenv("RAG_API_TOKEN", "")    # empty = auth disabled


app = FastAPI(
    title="CESS RAG Service",
    description="HTTP API around the CESS RAG library: ingest lectures, "
                "retrieve relevant chunks, manage per-session collections.",
    version="0.1.0",
)


# ---- Auth dependency --------------------------------------------------

def _check_auth(x_internal_token: str = Header(default="")) -> None:
    """Reject requests without the shared token. Skipped if RAG_API_TOKEN
    is unset (dev mode)."""
    if not API_TOKEN:
        return
    if x_internal_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Token")


# ---- Health -----------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "auth_enabled": bool(API_TOKEN)}


# ---- Ingestion --------------------------------------------------------

def _run_ingestion(job_id: str, req: IngestRequest) -> None:
    """Background worker: runs the pipeline and updates job state."""
    try:
        jobs.set_running(job_id)
        summary = ingest_files(
            files=req.files,
            collection_name=req.collection,
            data_root=req.data_root,
            use_florence=req.use_florence,
            lecture_num=req.lecture_num,
            lecture_title=req.lecture_title,
            replace=req.replace,
        )
        jobs.set_done(job_id, summary)
    except Exception:
        err = traceback.format_exc()
        jobs.set_failed(job_id, err)


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks,
           x_internal_token: str = Header(default="")):
    _check_auth(x_internal_token)

    # Validate that the files actually exist before queueing the job,
    # so the big project gets a clear error rather than a delayed failure.
    missing = [f for f in req.files if not Path(f).exists()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"File(s) not found on the RAG server: {missing}",
        )

    job_id = jobs.create_job()
    background_tasks.add_task(_run_ingestion, job_id, req)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingesting {len(req.files)} file(s) into '{req.collection}'.",
    )


@app.get("/ingest/{job_id}", response_model=JobStatusResponse)
def ingest_status(job_id: str, x_internal_token: str = Header(default="")):
    _check_auth(x_internal_token)
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return JobStatusResponse(**job)


# ---- Retrieval --------------------------------------------------------

# Retrievers are expensive to construct (they load the embedder + reranker
# on first query). Cache them per collection so repeat queries are fast.
_RETRIEVER_CACHE: dict = {}


def _get_retriever(collection: str, data_root: str) -> Retriever:
    key = (data_root, collection)
    if key not in _RETRIEVER_CACHE:
        chroma_dir = str(Path(data_root) / "chroma")
        _RETRIEVER_CACHE[key] = Retriever(
            db_path=chroma_dir,
            collection_name=collection,
        )
    return _RETRIEVER_CACHE[key]


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, x_internal_token: str = Header(default="")):
    _check_auth(x_internal_token)

    try:
        retriever = _get_retriever(req.collection, req.data_root)
        retriever.final_k = req.top_k
        hits = retriever.retrieve(req.query, lecture_num=req.lecture_num)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    results = [
        RetrievedChunk(
            chunk_id=h["chunk_id"],
            text=h["text"],
            metadata=h.get("metadata", {}),
            rerank_score=h.get("rerank_score"),
            fused_score=h.get("fused_score"),
        )
        for h in hits
    ]
    return RetrieveResponse(query=req.query, collection=req.collection, results=results)


# ---- Collection management -------------------------------------------

@app.delete("/collections/{collection}", response_model=DeleteCollectionResponse)
def delete_collection(collection: str, data_root: str = "data",
                      x_internal_token: str = Header(default="")):
    """Remove a session's collection from Chroma. Also deletes the
    matching BM25 pickle file. Called when the big project ends a session."""
    _check_auth(x_internal_token)

    chroma_dir = Path(data_root) / "chroma"
    try:
        store = VectorStore(db_path=str(chroma_dir), collection_name=collection)
        store._client.delete_collection(collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

    # Also remove the BM25 pickle if it exists
    bm25_path = chroma_dir / f"bm25_{collection}.pkl"
    if bm25_path.exists():
        bm25_path.unlink()

    # Drop the cached retriever for this collection so a future request
    # rebuilds it cleanly.
    _RETRIEVER_CACHE.pop((data_root, collection), None)

    return DeleteCollectionResponse(
        collection=collection,
        deleted=True,
        message="Collection and BM25 index removed.",
    )
