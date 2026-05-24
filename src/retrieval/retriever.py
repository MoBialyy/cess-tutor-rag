"""
Main retriever: orchestrates the full pipeline per report section 4.3.10.

Pipeline:
  1. Classify the query  -> alpha (sparse vs dense weight)
  2. Run BM25  + dense (INSTRUCTOR) in parallel, top-N each
  3. Fuse scores: final = alpha * sparse + (1 - alpha) * dense
  4. Cross-encoder rerank the fused top-N
  5. MMR diversify to final top-K

Typical defaults (matching report-aligned values):
  fetch_n = 20  (each retriever)
  rerank_n = 10
  final_k = 5

The retriever does NOT own the data — it borrows the Chroma collection
and BM25 index from the VectorStore. This keeps indexing and querying
decoupled: you ingest with `vector_store.py`, then query with `retriever.py`.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# Make sibling 'embedding' package importable when this file is run as a script
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embedding.embedder import Embedder
from embedding.vector_store import VectorStore

try:
    from .query_classifier import classify_query
    from .bm25_retriever import BM25Retriever
    from .reranker import Reranker
    from .mmr import mmr_select
except ImportError:
    from query_classifier import classify_query
    from bm25_retriever import BM25Retriever
    from reranker import Reranker
    from mmr import mmr_select


def _minmax_normalize(scores: List[float]) -> List[float]:
    """Scale a list of scores to [0, 1]. Used for fusing BM25 (raw) with
    dense (cosine, already roughly [0, 1])."""
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return [1.0] * len(scores)
    return ((arr - lo) / (hi - lo)).tolist()


class Retriever:
    def __init__(
        self,
        db_path: str = "data/chroma",
        collection_name: str = "default_course",
        fetch_n: int = 20,
        rerank_n: int = 10,
        final_k: int = 5,
        mmr_lambda: float = 0.6,
    ):
        self.fetch_n = fetch_n
        self.rerank_n = rerank_n
        self.final_k = final_k
        self.mmr_lambda = mmr_lambda

        # Shared embedder (used by VectorStore for queries and by us for MMR)
        self.embedder = Embedder()
        self.vector_store = VectorStore(
            db_path=db_path,
            collection_name=collection_name,
            embedder=self.embedder,
        )

        # BM25 index lives next to the Chroma data
        bm25_path = str(Path(db_path) / f"bm25_{collection_name}.pkl")
        self.bm25 = BM25Retriever(index_path=bm25_path)

        self.reranker = Reranker()

    # ---- the main entry point ---------------------------------------

    def retrieve(self, query: str, verbose: bool = False,
                 lecture_num: Optional[int] = None) -> List[Dict]:
        """Run the full retrieval pipeline and return final top-K chunks."""

        # 1. Classify
        cls = classify_query(query)
        alpha = cls["alpha"]
        if verbose:
            print(f"[retriever] query type: {cls['type']} (alpha={alpha})")

        # 2a. Sparse retrieval
        bm25_hits = self.bm25.search(query, top_k=self.fetch_n, lecture_num=lecture_num)

        # 2b. Dense retrieval (via the existing VectorStore)
        dense_hits = self.vector_store.query(query, top_k=self.fetch_n, lecture_num=lecture_num)

        if verbose:
            print(f"[retriever] bm25 hits: {len(bm25_hits)}, dense hits: {len(dense_hits)}")

        # 3. Fuse scores
        fused = self._fuse_scores(bm25_hits, dense_hits, alpha)
        if verbose:
            print(f"[retriever] fused candidates: {len(fused)}")

        # Trim to rerank_n before paying the cross-encoder cost
        fused = fused[: max(self.rerank_n, self.final_k * 2)]

        # 4. Cross-encoder rerank
        reranked = self.reranker.rerank(query, fused, top_k=self.rerank_n)
        if verbose:
            print(f"[retriever] reranked to top {len(reranked)}")

        # 5. MMR diversification (needs embeddings on each candidate)
        self._attach_chunk_embeddings(reranked)
        final = mmr_select(reranked, top_k=self.final_k, lambda_param=self.mmr_lambda)

        return final

    # ---- internal helpers -------------------------------------------

    def _fuse_scores(
        self,
        bm25_hits: List[Dict],
        dense_hits: List[Dict],
        alpha: float,
    ) -> List[Dict]:
        """Combine BM25 and dense results by weighted score fusion."""

        # Index by chunk_id so we can merge the two ranking lists
        bm25_scores = {h["chunk_id"]: h["score"] for h in bm25_hits}
        dense_by_id = {h["chunk_id"]: h for h in dense_hits}

        # Dense distance: smaller = more similar. Convert to similarity = 1 - distance.
        dense_scores = {
            cid: 1.0 - h["distance"] for cid, h in dense_by_id.items()
        }

        # Normalize each score list independently so they live on the same scale
        bm25_keys = list(bm25_scores.keys())
        dense_keys = list(dense_scores.keys())
        bm25_norm = dict(zip(bm25_keys, _minmax_normalize([bm25_scores[k] for k in bm25_keys])))
        dense_norm = dict(zip(dense_keys, _minmax_normalize([dense_scores[k] for k in dense_keys])))

        all_ids = set(bm25_norm) | set(dense_norm)
        fused = []
        for cid in all_ids:
            s_sparse = bm25_norm.get(cid, 0.0)
            s_dense  = dense_norm.get(cid, 0.0)
            score = alpha * s_sparse + (1 - alpha) * s_dense

            # Prefer the dense entry for the full document/metadata payload
            # since it has Chroma metadata. Fall back to BM25 entry if missing.
            base = dense_by_id.get(cid)
            if base is None:
                # Re-fetch from BM25 results
                base_hit = next((h for h in bm25_hits if h["chunk_id"] == cid), None)
                base = {
                    "chunk_id": cid,
                    "text": base_hit["text"] if base_hit else "",
                    "metadata": {},
                }
            fused.append({
                "chunk_id": cid,
                "text": base["text"],
                "metadata": base.get("metadata", {}),
                "fused_score": score,
                "sparse_score": s_sparse,
                "dense_score": s_dense,
            })

        fused.sort(key=lambda c: c["fused_score"], reverse=True)
        return fused

    def _attach_chunk_embeddings(self, candidates: List[Dict]) -> None:
        """Re-embed each candidate chunk's text. Needed for MMR.
        For ~10 chunks this is fast even on CPU."""
        if not candidates:
            return
        texts = [c["text"] for c in candidates]
        embs = self.embedder.embed_chunks(texts)
        for c, e in zip(candidates, embs):
            c["embedding"] = e


# ---- CLI -------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full retrieval pipeline: hybrid (BM25 + dense) + rerank + MMR"
    )
    parser.add_argument("--db-path", default="data/chroma")
    parser.add_argument("--collection", required=True,
                        help="Chroma collection name, e.g. 'nlp_course'")
    parser.add_argument("--query", required=True, help="Student question")
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--lecture-num", type=int, default=None,
                        help="Optional lecture filter (only return chunks from this lecture)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    retr = Retriever(
        db_path=args.db_path,
        collection_name=args.collection,
        final_k=args.final_k,
    )
    results = retr.retrieve(args.query, verbose=args.verbose, lecture_num=args.lecture_num)

    print(f"\n=== Top {len(results)} results ===\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"  chunk_id:     {r['chunk_id']}")
        print(f"  metadata:     {r['metadata']}")
        print(f"  rerank_score: {r.get('rerank_score', 0):.4f}")
        print(f"  fused_score:  {r.get('fused_score', 0):.4f}")
        snippet = r['text'].replace('\n', ' ')[:240]
        print(f"  text:         {snippet}{'...' if len(r['text']) > 240 else ''}")
        print()
