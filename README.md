# CESS Tutor RAG

A Retrieval-Augmented Generation (RAG) pipeline that turns university lecture
materials (PDF/PPTX) into a queryable knowledge base. Built as part of a
graduation project at the Computer Engineering & Software Systems (CESS)
department, Ain Shams University.

> **Note** — This repository contains only the **RAG pipeline and its HTTP API**.
> It is not a full application with a user interface. It is designed to be
> consumed as a microservice by a larger web application that handles users,
> sessions, file uploads, and the LLM tutor itself. The pipeline accepts
> uploaded files and returns ranked text chunks; everything around it is
> outside the scope of this repo.

## What this project does

Given one or more lecture files (PDF, PPTX), the pipeline:

1. Extracts text and structures it into normalized Markdown
2. Captions images (diagrams, code screenshots, tables) using a vision-language
   model so that visual content is searchable as text
3. Splits the result into semantically coherent chunks
4. Embeds chunks with an instruction-aware model
5. Stores them in a local vector database alongside a sparse index
6. Serves retrieval over HTTP, returning top-K chunks for any natural-language
   question

The end result is that a downstream LLM tutor can ask the API for the most
relevant slides for a given question, and ground its answer in the user's
actual course materials.

## Architecture

```
PDF / PPTX
    │
    ▼
[1] Preprocessing      ← PyMuPDF, python-pptx, Florence-2, Tesseract OCR
    │
    ▼
   .md (normalized markdown with slide headers, bullets, image captions)
    │
    ▼
[2] Chunking           ← RecursiveCharacterTextSplitter + INSTRUCTOR tokenizer
    │
    ▼
   .jsonl (chunks of ~500 tokens, 75-token overlap)
    │
    ▼
[3] Embedding & Storage ← INSTRUCTOR-base (instruction-aware)
    │                     ChromaDB (dense) + BM25 pickle (sparse)
    ▼
[4] Retrieval          ← Hybrid (BM25 + dense) + cross-encoder rerank + MMR
    │
    ▼
[5] HTTP API           ← FastAPI service with async ingestion jobs
```

---

## Process details

### 1 — Preprocessing

Turns lecture files into normalized Markdown with explicit slide boundaries
(`## Slide N: Title`), bullets, and inline captions for embedded images.

**Tech:** PyMuPDF (PDF parsing), python-pptx (PowerPoint parsing),
[Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base) (vision-language
captioning), Tesseract OCR (printed text recognition).

**Problems solved:**

- **PDFs lose structure on export.** PowerPoint slides exported to PDF become
  flat positioned glyphs, with no notion of slide boundaries, bullets, or
  headings. Heuristics on font size, position, and bullet glyphs reconstruct
  the original pedagogical structure.
- **Visual content is not searchable.** Diagrams, code screenshots, and
  comparison tables are images, invisible to text-only retrieval. The pipeline
  routes each image through either Tesseract (for slides that are essentially
  pages of typed prose) or Florence-2 (for everything else), then attaches a
  short domain-specific label such as *"A code snippet"*, *"A comparison or
  reference table"*, or *"A diagram or flowchart illustrating the concept"*.
- **PowerPoint template artifacts pollute output.** Logos, decorative
  backgrounds, and recurring footer images appear on every slide. The pipeline
  hashes images on slides 1–2 (template-defining slides) and treats matches on
  later slides as boilerplate to skip.
- **Noisy decorations slip through.** A second layer of keyword-based filtering
  catches Florence-2 captions like *"a stack of books"* or *"a question mark"*
  and drops them.

### 2 — Chunking

Splits the markdown into chunks of approximately 450–550 tokens (target 500)
with 75-token overlap.

**Tech:** LangChain's `RecursiveCharacterTextSplitter` with a hierarchy of
semantic separators (slide headers → paragraphs → sentences → words),
token-counted using the INSTRUCTOR tokenizer.

**Problems solved:**

- **Fixed-size chunking fragments concepts.** A naive 512-character split
  cuts mid-sentence, separating definitions from their examples. The
  hierarchical separator list prioritizes slide boundaries, then paragraph
  breaks, falling back to finer separators only when size constraints force it.
- **Token counts must match the embedding model.** Character counts mislead;
  Chinese characters, code, and English prose tokenize to wildly different
  lengths. The chunker uses the same tokenizer the embedding model will use
  later, so chunk sizes are always accurate against the model's 512-token
  context window.
- **Cross-boundary continuity.** Concepts that span chunk boundaries (a
  definition at the end of one chunk, its example at the start of the next)
  would otherwise be retrieved in isolation. A 75-token overlap preserves
  conceptual continuity.

### 3 — Embedding & Storage

Each chunk is embedded as a 768-dimensional vector and stored alongside a
parallel sparse index.

**Tech:** [INSTRUCTOR-base](https://huggingface.co/hkunlp/instructor-base) via
sentence-transformers, [ChromaDB](https://www.trychroma.com/) for the vector
store (persistent on-disk SQLite + HNSW index), `rank-bm25` for the sparse
keyword index.

**Problems solved:**

- **Naive embeddings don't capture query intent.** Student questions like
  *"explain how X works"* differ in intent from declarative course content.
  INSTRUCTOR prepends task instructions to both the indexed text and the
  query, biasing the embedding space toward question-answering retrieval
  rather than literal text similarity.
- **Local-first deployment.** ChromaDB runs as an embedded library, with no
  separate database server. The entire vector store is a folder of files on
  disk, trivially backed up or moved between machines.
- **Hybrid retrieval needs sparse and dense in sync.** The BM25 pickle is
  rebuilt automatically whenever chunks are added, so the two indexes never
  drift.

### 4 — Retrieval

Given a query, returns the top-K most relevant chunks via a multi-stage
hybrid pipeline.

**Tech:**

- Query classifier (lightweight regex + keyword heuristics) routes queries to
  *code*, *conceptual*, or *definitional* intent
- BM25 sparse retrieval (rank-bm25) for lexical matches
- INSTRUCTOR-base dense retrieval via ChromaDB for semantic matches
- Weighted score fusion (`alpha = 0.6 / 0.4 / 0.5` depending on intent)
- Cross-encoder reranker
  ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2))
  re-scores top candidates with query/document joint encoding
- Maximal Marginal Relevance (MMR) diversifies the final top-K to avoid
  near-duplicate chunks

**Problems solved:**

- **Dense retrieval misses exact terms.** Queries like *"`re.findall`"* or
  *"NER"* depend on exact lexical match. Dense models can confuse closely
  related technical concepts (e.g. returning a tiny "Removing Stop Words"
  header for a "what is stemming" query). BM25 cleanly recovers these cases.
- **Sparse retrieval misses paraphrases.** A student asking *"how do I split
  text into words?"* shares no terms with a slide titled *"Tokenization"*.
  Dense retrieval catches this.
- **Top-K from either alone is noisy.** Reranking with a cross-encoder
  evaluates query and chunk *jointly*, catching mismatches that bi-encoder
  retrieval misses.
- **Top-K can be near-duplicates.** Without MMR, three of the top-5 might be
  variations of the same slide. MMR ensures the final list spans different
  aspects of the answer.
- **Per-lecture scoping.** When a user is studying a single lecture, retrieval
  should not bleed in chunks from other lectures in the same course. Each
  chunk is stamped with a `lecture_num` metadata field, and queries can be
  filtered to a single lecture.

### 5 — HTTP API

A FastAPI service exposes the pipeline so the larger web application can call
it over HTTP.

**Tech:** FastAPI, Uvicorn, Pydantic. Optional shared-secret authentication
via the `X-Internal-Token` header.

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/health` | Service status |
| `POST`   | `/ingest` | Kick off background ingestion of one or more files |
| `GET`    | `/ingest/{job_id}` | Poll status of an ingestion job |
| `POST`   | `/retrieve` | Run hybrid retrieval against a collection |
| `DELETE` | `/collections/{collection}` | Remove a collection and its BM25 index |

**Problems solved:**

- **Ingestion is slow** (a typical lecture takes 1–3 minutes due to image
  processing). Long-running HTTP requests are fragile. Ingestion runs as a
  background task and the client polls a job-status endpoint.
- **Per-user / per-session isolation.** The big project passes a `data_root`
  path on every request, so each user's files live in their own folder
  (`data/users/<user_id>/...`). Collections are scoped further by name
  (e.g. one collection per course or per session).
- **Upsert semantics for re-uploaded lectures.** Passing `replace: true` in
  the ingest request deletes existing chunks for that `lecture_num` before
  adding the new ones, so a user updating their copy of a lecture cleanly
  replaces the old content.

---

## Repository layout

```
cess-tutor-rag/
├── src/
│   ├── preprocessing/      # PDF/PPTX extractors, Florence, Tesseract, filters
│   ├── chunking/           # constrained semantic chunker + token counter
│   ├── embedding/          # INSTRUCTOR embedder + ChromaDB vector store
│   ├── retrieval/          # query classifier, BM25, reranker, MMR, orchestrator
│   └── pipeline.py         # end-to-end orchestrator (preprocess → chunk → ingest)
├── service/
│   ├── main.py             # FastAPI app
│   ├── models.py           # Pydantic request/response schemas
│   └── jobs.py             # in-memory async job tracker
├── data/                   # runtime data (gitignored)
│   ├── uploads/            # user uploads
│   ├── processed/          # intermediate .md
│   ├── chunks/             # intermediate .jsonl
│   └── chroma/             # vector store + BM25 indexes
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## How to run

### 1. Install Tesseract OCR

Tesseract is a system binary, not a Python package — install it before
anything else.

**Windows:** Download and run the installer from
[UB-Mannheim/tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki).
Latest stable as of writing: `tesseract-ocr-w64-setup-5.5.0.20241111.exe`.
Default install path is usually `C:\Program Files\Tesseract-OCR\tesseract.exe`.

**macOS:** `brew install tesseract`

**Linux:** `sudo apt install tesseract-ocr` (Debian/Ubuntu)

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/<your-username>/cess-tutor-rag.git
cd cess-tutor-rag
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and edit it:

```
# Windows-only: path to the Tesseract executable
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Optional shared-secret for HTTP API auth (leave empty to disable)
RAG_API_TOKEN=
```

On macOS/Linux, `TESSERACT_CMD` can be omitted if `tesseract` is on your PATH.

### 4. First-time model downloads

The first run downloads three Hugging Face models (totaling ~1 GB):

- Florence-2-base (~463 MB)
- INSTRUCTOR-base (~440 MB)
- ms-marco-MiniLM-L-6-v2 (~80 MB)

These are cached in `~/.cache/huggingface/hub/` (or `C:\Users\<you>\.cache\huggingface\hub\`
on Windows) and reused across runs.

### 5. Run the service

```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) in a browser to
explore the interactive API documentation.

### 6. Try it

Ingest a lecture:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["/absolute/path/to/Lecture_3.pdf"],
    "collection": "ethics_course",
    "lecture_num": 3,
    "lecture_title": "Space Shuttle Challenger Case Study"
  }'
```

The response includes a `job_id`. Poll for completion:

```bash
curl http://localhost:8000/ingest/<job_id>
```

Once status is `done`, retrieve:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what caused the challenger explosion?",
    "collection": "ethics_course",
    "top_k": 5
  }'
```

To restrict to a single lecture, add `"lecture_num": 3` to the retrieve body.

---

## Using as a Python library

The pipeline can also be imported directly without the HTTP layer:

```python
from src.pipeline import ingest_files
from src.retrieval.retriever import Retriever

# Ingest
ingest_files(
    files=["data/uploads/Lecture_3.pdf"],
    collection_name="ethics_course",
    data_root="data",
    lecture_num=3,
    lecture_title="Space Shuttle Challenger Case Study",
)

# Retrieve
retr = Retriever(db_path="data/chroma", collection_name="ethics_course")
results = retr.retrieve("what caused the challenger explosion?")
```

---

## Tech summary

| Layer | Library / model |
|---|---|
| PDF parsing | PyMuPDF |
| PPTX parsing | python-pptx |
| OCR | Tesseract (via pytesseract) |
| Image captioning | Florence-2-base |
| Chunking | langchain-text-splitters |
| Token counting | INSTRUCTOR-XL tokenizer (via transformers) |
| Embeddings | INSTRUCTOR-base (sentence-transformers) |
| Vector store | ChromaDB |
| Sparse retrieval | rank-bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Web framework | FastAPI |
| ASGI server | Uvicorn |

---

## Context

This pipeline is one component of a larger graduation project at the
Computer Engineering & Software Systems (CESS) department, Ain Shams
University. The full project is an LLM-powered tutoring assistant for
university courses, combining this RAG pipeline with an LLM, pedagogical
guardrails, and a personalization engine. This repository contains only the
retrieval-augmented generation layer.
