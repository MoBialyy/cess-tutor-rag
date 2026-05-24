"""
Pydantic request/response schemas for the RAG microservice.

These define the JSON shape of every endpoint's input and output. FastAPI
uses them for automatic validation and to generate the /docs page.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---- /ingest -----------------------------------------------------------

class IngestRequest(BaseModel):
    files: List[str] = Field(
        ...,
        description="Absolute paths to uploaded PDF or PPTX files on the RAG server's filesystem."
    )
    collection: str = Field(
        ...,
        description="Collection name to store the chunks in, e.g. 'session_abc123'.",
    )
    data_root: str = Field(
        "data",
        description="Root directory for this user's data. Big project should pass 'data/users/<user_id>'.",
    )
    use_florence: bool = Field(
        True,
        description="Run Florence-2 image processing (slow but recommended). "
                    "Set false for fast text-only ingestion.",
    )
    lecture_num: Optional[int] = Field(
        None,
        description="Lecture number (used for filtering retrieval to a single lecture).",
    )
    lecture_title: str = Field(
        "",
        description="Human-readable lecture title (used for citations).",
    )
    replace: bool = Field(
        False,
        description="If true and lecture_num is set, delete any existing chunks "
                    "with that lecture_num before ingesting (i.e., upsert).",
    )


class IngestResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="One of: queued, running, done, failed")
    message: str = ""


# ---- /ingest/{job_id} --------------------------------------------------

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""
    summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Pipeline summary dict (only set when status == 'done').",
    )


# ---- /retrieve ---------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str
    collection: str
    data_root: str = "data"
    top_k: int = 5
    lecture_num: Optional[int] = Field(
        None,
        description="If set, only retrieve chunks from this lecture within the collection.",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    rerank_score: Optional[float] = None
    fused_score: Optional[float] = None


class RetrieveResponse(BaseModel):
    query: str
    collection: str
    results: List[RetrievedChunk]


# ---- /collections/{name} ----------------------------------------------

class DeleteCollectionResponse(BaseModel):
    collection: str
    deleted: bool
    message: str = ""
