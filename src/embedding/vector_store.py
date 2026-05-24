"""
ChromaDB-backed vector store for CESS course chunks.

One collection per course (e.g. "nlp_course", "ethics_course") — not per
lecture. Each entry stores:
  - id:        chunk_id from the JSONL
  - embedding: 768-d INSTRUCTOR-base vector (cosine space)
  - document:  the chunk text itself
  - metadata:  source, slide_range, topic, tokens

Two main operations:
  - add_chunks(): ingest a JSONL file produced by the chunker
  - query():     embed a student question and retrieve the top-k chunks

The DB lives on disk at --db-path (default: data/chroma). Chroma creates
the folder on first run; no server, no extra setup. Same idea as SQLite.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

try:
    from .embedder import Embedder    # when imported as embedding.vector_store
except ImportError:
    from embedder import Embedder     # when run as a script


class VectorStore:
    def __init__(
        self,
        db_path: str = "data/chroma",
        collection_name: str = "default_course",
        embedder: Optional[Embedder] = None,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()

        # Persistent client = on-disk SQLite + binary index files.
        # anonymized_telemetry=False stops Chroma from phoning home.
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create_collection: idempotent, safe to call repeatedly.
        # metadata={"hnsw:space": "cosine"} matches our normalized embeddings.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- ingestion ---------------------------------------------------

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add a list of chunk dicts (as produced by chunker.py) to the
        collection. Skips chunks whose chunk_id is already present so
        re-ingesting the same lecture is safe.

        Returns the number of newly added chunks. BM25 index is always
        rebuilt after this call (even if 0 new chunks were added) to
        ensure the hybrid retriever has a working index.
        """
        if not chunks:
            return 0

        # Skip duplicates: ask Chroma which ids it already has
        all_ids = [c["chunk_id"] for c in chunks]
        existing = set(self._collection.get(ids=all_ids).get("ids", []))
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

        if not new_chunks:
            print(f"[vector_store] all {len(chunks)} chunks already indexed.")
            # Still rebuild BM25 -- it may not exist on disk yet
            self.rebuild_bm25()
            return 0

        if existing:
            print(f"[vector_store] skipping {len(existing)} already-indexed chunks; "
                  f"embedding {len(new_chunks)} new ones.")

        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.embed_chunks(texts)

        self._collection.add(
            ids=[c["chunk_id"] for c in new_chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{
                "source": c.get("source", ""),
                "slide_range": c.get("slide_range", ""),
                "topic": c.get("topic", ""),
                "tokens": c.get("tokens", 0),
                "lecture_num": c.get("lecture_num") if c.get("lecture_num") is not None else -1,
                "lecture_title": c.get("lecture_title", ""),
            } for c in new_chunks],
        )

        # Always rebuild BM25 alongside Chroma so the hybrid retriever
        # finds it. We rebuild from the full corpus, not just new chunks,
        # because BM25 needs corpus-level statistics.
        self.rebuild_bm25()

        return len(new_chunks)
    
    def delete_by_lecture(self, lecture_num: int) -> int:
        """Delete all chunks in the collection with the given lecture_num.
        Also rebuilds the BM25 index so it stays in sync.
        Returns the number of chunks deleted."""
        if lecture_num is None:
            return 0

        # Find matching chunks first so we can report the count
        matches = self._collection.get(where={"lecture_num": lecture_num})
        count = len(matches.get("ids", []))

        if count == 0:
            return 0

        self._collection.delete(where={"lecture_num": lecture_num})
        print(f"[vector_store] deleted {count} chunks with lecture_num={lecture_num}")

        # Rebuild BM25 from whatever's left
        self.rebuild_bm25()
        return count

    def rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from every chunk currently in the
        collection and persist it next to the Chroma data. Safe to call
        any time; callable from the CLI via --rebuild-bm25."""
        try:
            # Lazy import so vector_store doesn't hard-depend on retrieval
            import sys
            _src_dir = Path(__file__).resolve().parent.parent
            if str(_src_dir) not in sys.path:
                sys.path.insert(0, str(_src_dir))
            from retrieval.bm25_retriever import BM25Retriever
        except ImportError:
            print("[vector_store] retrieval module not found; skipping BM25 rebuild.")
            return

        all_data = self._collection.get(include=["documents", "metadatas"])
        chunks = []
        for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            lecture_num = meta.get("lecture_num") if meta else None
            # -1 was our sentinel for "no lecture number"
            if lecture_num == -1:
                lecture_num = None
            chunks.append({"chunk_id": cid, "text": doc, "lecture_num": lecture_num})
        bm25_path = str(Path(self.db_path) / f"bm25_{self.collection_name}.pkl")
        retriever = BM25Retriever(index_path=bm25_path)
        retriever.build(chunks)
        print(f"[vector_store] BM25 index rebuilt ({len(chunks)} chunks) at {bm25_path}")

    def add_jsonl(self, jsonl_path: str) -> int:
        """Convenience: load a chunker JSONL file and ingest it."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(jsonl_path)

        chunks = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        print(f"[vector_store] loaded {len(chunks)} chunks from {jsonl_path}")
        return self.add_chunks(chunks)

    # ---- retrieval ---------------------------------------------------

    def query(self, question: str, top_k: int = 5,
              lecture_num: Optional[int] = None) -> List[Dict]:
        """
        Embed a student question and return the top-k most relevant chunks.
        Each result has: chunk_id, text, metadata, distance (lower = more similar).
        """
        q_emb = self.embedder.embed_query(question)
        where_filter = {"lecture_num": lecture_num} if lecture_num is not None else None
        res = self._collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        # Chroma returns parallel lists wrapped in a [batch] dim; unwrap.
        ids       = res.get("ids", [[]])[0]
        docs      = res.get("documents", [[]])[0]
        metas     = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        return [
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]

    # ---- inspection --------------------------------------------------

    def count(self) -> int:
        return self._collection.count()


# ---- CLI -------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a chunker JSONL into a Chroma collection, "
                    "or query an existing one."
    )
    parser.add_argument("input", nargs="?",
                        help="Path to chunker JSONL to ingest (omit if querying)")
    parser.add_argument("--db-path", default="data/chroma",
                        help="Chroma persistent path (default: data/chroma)")
    parser.add_argument("--collection", required=True,
                        help="Collection name, e.g. 'nlp_course'")
    parser.add_argument("--query", default=None,
                        help="Run a retrieval query against the collection")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return when --query is set")
    parser.add_argument("--rebuild-bm25", action="store_true",
                        help="Force a BM25 index rebuild from the existing collection")
    args = parser.parse_args()

    store = VectorStore(db_path=args.db_path, collection_name=args.collection)

    if args.rebuild_bm25:
        store.rebuild_bm25()

    if args.input:
        added = store.add_jsonl(args.input)
        print(f"[vector_store] added {added} chunks. "
              f"Collection '{args.collection}' now has {store.count()} chunks total.")

    if args.query:
        print(f"\n[vector_store] querying: {args.query!r}\n")
        results = store.query(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"--- Result {i}  (distance={r['distance']:.4f}) ---")
            print(f"  chunk_id: {r['chunk_id']}")
            print(f"  metadata: {r['metadata']}")
            print(f"  text:     {r['text'][:240]}{'...' if len(r['text']) > 240 else ''}")
            print()

    if not args.input and not args.query and not args.rebuild_bm25:
        print(f"[vector_store] Collection '{args.collection}' has "
              f"{store.count()} chunks. Pass an input JSONL to add more, "
              f"or --query to retrieve, or --rebuild-bm25 to refresh BM25.")
