"""
Cross-encoder reranker for stage-1 retrieval refinement.

Per report section 5.4.6, the bi-encoder (INSTRUCTOR) and BM25 produce
top-N candidates based on independent scoring. A cross-encoder then
re-scores each (query, chunk) pair *jointly*, capturing fine-grained
semantic interaction that the bi-encoder misses.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 80MB, ~50ms per pair on CPU
  - Trained on MS MARCO passage ranking
  - Standard choice for English retrieval reranking

Loaded lazily so the 80MB download only happens on first query.
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder


_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_id: str = _MODEL_ID, device: str = None):
        self.model_id = model_id
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            print(f"[reranker] loading {self.model_id}...", flush=True)
            self._model = CrossEncoder(self.model_id, device=self.device)
            print(f"[reranker] ready", flush=True)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Re-score (query, chunk) pairs with a cross-encoder and return the
        top-k. Each candidate dict must have 'text'; the rerank score is
        attached as 'rerank_score' on each returned dict.
        """
        if not candidates:
            return []
        self._ensure_model()

        pairs = [[query, c["text"]] for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort descending
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
