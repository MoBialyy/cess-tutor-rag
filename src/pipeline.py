"""
End-to-end RAG pipeline orchestrator.

Takes one or more uploaded files (PDF or PPTX), runs the full
preprocess -> chunk -> embed/store pipeline, and adds the resulting
chunks to a Chroma collection.

Intermediate artifacts are written to disk for debugging:
  - data/processed/<stem>.md    (preprocessed markdown)
  - data/chunks/<stem>.jsonl    (chunked JSONL)

The final vector store lives in data/chroma/ alongside the BM25 indexes.
For the eventual multi-user web app, each user gets their own data/
directory (data/users/<user_id>/...) with the same structure.

Usage:
  CLI:
    python src/pipeline.py file1.pdf file2.pptx --collection nlp_course

  Python:
    from pipeline import ingest_files
    ingest_files(
        files=["uploads/L1.pdf", "uploads/L2.pptx"],
        collection_name="nlp_course",
        data_root="data",
    )
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

# Make sibling packages importable when run as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from preprocessing.preprocessor import Preprocessor
from chunking.chunker import Chunker
from embedding.vector_store import VectorStore


def ingest_files(
    files: List[str],
    collection_name: str,
    data_root: str = "data",
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    use_florence: bool = True,
    lecture_num: Optional[int] = None,
    lecture_title: str = "",
    replace: bool = False,
) -> Dict:
    """
    Run the full pipeline on a batch of uploaded files and add the chunks
    to the given Chroma collection.

    Returns a summary dict:
      {
        "collection": "nlp_course",
        "files_processed": 2,
        "chunks_added": 24,
        "total_in_collection": 24,
        "per_file": [
          {"file": "L1.pdf", "md": ".../L1.md", "jsonl": ".../L1.jsonl", "chunks": 12},
          ...
        ],
      }
    """
    data = Path(data_root)
    processed_dir = data / "processed"
    chunks_dir    = data / "chunks"
    chroma_dir    = data / "chroma"
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # One-time setup, reused across all files in this batch
    preprocessor = Preprocessor(use_florence=use_florence)
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = VectorStore(db_path=str(chroma_dir), collection_name=collection_name)
    # If replace mode + a lecture number is given, wipe any existing chunks
    # for that lecture before ingesting the new ones.
    if replace and lecture_num is not None:
        deleted = store.delete_by_lecture(lecture_num)
        if deleted > 0:
            print(f"[pipeline] replaced existing lecture {lecture_num}: removed {deleted} chunks")

    per_file = []
    total_added = 0

    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"[pipeline] SKIP {f} — file not found")
            continue

        stem = path.stem
        md_path = processed_dir / f"{stem}.md"
        jsonl_path = chunks_dir / f"{stem}.jsonl"

        print(f"\n[pipeline] === Processing {path.name} ===", flush=True)

        # 1. Preprocess
        print(f"[pipeline] preprocessing -> {md_path}", flush=True)
        preprocessor.process(str(path), output_path=str(md_path))

        # 2. Chunk
        print(f"[pipeline] chunking -> {jsonl_path}", flush=True)
        chunks = chunker.chunk_file(
            str(md_path),
            source_name=stem,
            lecture_num=lecture_num,
            lecture_title=lecture_title,
        )
        chunker.write_jsonl(chunks, str(jsonl_path))
        print(f"[pipeline]   {len(chunks)} chunks", flush=True)

        # 3. Ingest into vector store
        print(f"[pipeline] ingesting into '{collection_name}'", flush=True)
        added = store.add_chunks(chunks)
        total_added += added

        per_file.append({
            "file": path.name,
            "md": str(md_path),
            "jsonl": str(jsonl_path),
            "chunks": len(chunks),
            "newly_added": added,
        })

    summary = {
        "collection": collection_name,
        "files_processed": len(per_file),
        "chunks_added": total_added,
        "total_in_collection": store.count(),
        "per_file": per_file,
    }

    print(f"\n[pipeline] === Done ===")
    print(f"[pipeline]   {summary['files_processed']} files processed")
    print(f"[pipeline]   {summary['chunks_added']} new chunks added")
    print(f"[pipeline]   '{collection_name}' now has {summary['total_in_collection']} chunks total")
    return summary


# ---- CLI -------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full RAG pipeline on a batch of uploaded files."
    )
    parser.add_argument("files", nargs="+",
                        help="One or more PDF or PPTX files to ingest")
    parser.add_argument("--collection", required=True,
                        help="Chroma collection name, e.g. 'nlp_course'")
    parser.add_argument("--data-root", default="data",
                        help="Root data directory (default: data)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=75)
    parser.add_argument("--no-images", action="store_true",
                        help="Skip Florence-2 image processing")
    parser.add_argument("--lecture-num", type=int, default=None,
                        help="Lecture number to stamp on every chunk")
    parser.add_argument("--lecture-title", default="",
                        help="Lecture title to stamp on every chunk")
    parser.add_argument("--replace", action="store_true",
                        help="If a lecture with this lecture_num already exists, delete it first.")
    args = parser.parse_args()

    ingest_files(
        files=args.files,
        collection_name=args.collection,
        data_root=args.data_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_florence=not args.no_images,
        lecture_num=args.lecture_num,
        lecture_title=args.lecture_title,
        replace=args.replace,
    )