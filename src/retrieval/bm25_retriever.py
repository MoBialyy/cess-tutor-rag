"""
BM25 sparse retriever for CESS chunks.

BM25 ranks documents by term overlap with the query, normalized for
document length and with diminishing returns on repeated terms. It is
strong on:
  - exact term matches (function names, technical jargon)
  - code snippets where dense embeddings struggle with syntax
  - acronyms and specific identifiers (NER, POS, etc.)

We persist the index as a pickle file alongside the Chroma collection
so it survives restarts. The index is small (a few hundred KB for
~100 chunks) and rebuilds in milliseconds when chunks are added.
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


# Simple word tokenizer. Lowercase, splits on non-word characters,
# keeps underscores so identifiers like "word_tokenize" stay intact.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self._bm25: Optional[BM25Okapi] = None
        # Parallel lists: chunk_ids[i] corresponds to corpus_tokens[i]
        self._chunk_ids: List[str] = []
        self._chunk_texts: List[str] = []
        self._corpus_tokens: List[List[str]] = []
        self._lecture_nums: List[Optional[int]] = []

        if index_path and Path(index_path).exists():
            self._load()

    # ---- index building ---------------------------------------------

    def build(self, chunks: List[Dict]) -> None:
        """
        Build a fresh BM25 index from a list of chunk dicts (chunk_id, text,
        and optionally lecture_num). Replaces any existing index. Persists
        if index_path is set.
        """
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._chunk_texts = [c["text"] for c in chunks]
        self._lecture_nums = [c.get("lecture_num") for c in chunks]
        self._corpus_tokens = [_tokenize(t) for t in self._chunk_texts]

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

        if self.index_path:
            self._save()

    # ---- retrieval --------------------------------------------------

    def search(self, query: str, top_k: int = 20,
               lecture_num: Optional[int] = None) -> List[Dict]:
        """
        Return the top-k chunks by BM25 score.
        Each result: {chunk_id, text, score} (score is raw BM25, higher = better).
        """
        if not self._bm25 or not self._corpus_tokens:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)

        # Top-k indices by score, descending
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "chunk_id": self._chunk_ids[i],
                "text": self._chunk_texts[i],
                "score": float(scores[i]),
            }
            for i in ranked
            if scores[i] > 0
            and (lecture_num is None or self._lecture_nums[i] == lecture_num)
        ]

    # ---- persistence ------------------------------------------------

    def _save(self) -> None:
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "chunk_texts": self._chunk_texts,
                "corpus_tokens": self._corpus_tokens,
                "lecture_nums": self._lecture_nums,
            }, f)

    def _load(self) -> None:
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._chunk_texts = data["chunk_texts"]
        self._corpus_tokens = data["corpus_tokens"]
        self._lecture_nums = data.get("lecture_nums", [None] * len(self._chunk_ids))
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def __len__(self) -> int:
        return len(self._chunk_ids)
