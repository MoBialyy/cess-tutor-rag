"""
INSTRUCTOR-base embedder.

INSTRUCTOR is an instruction-aware embedding model: you prepend a task
instruction to every input, and the model produces embeddings that reflect
both the content AND the task intent. This is especially useful for
course-tutoring retrieval where queries are framed as questions ("explain
how X works", "what is Y") rather than keyword searches.

We use INSTRUCTOR-base (110M params, 768-dim output). Loaded lazily so
the 440MB model is only fetched when you actually embed something.

The model originally shipped with the `InstructorEmbedding` package, which
is now unmaintained. The recommended path is to load it through
`sentence-transformers` directly, which supports the same instruction
prefixing natively.
"""

from typing import List, Iterable
from sentence_transformers import SentenceTransformer
import numpy as np


# Task-specific instructions. INSTRUCTOR uses these to bias the embedding
# space toward the right notion of similarity.
INDEX_INSTRUCTION = (
    "Represent the computer engineering course material for retrieval:"
)
QUERY_INSTRUCTION = (
    "Represent the student question for retrieving relevant course material:"
)

_MODEL_ID = "hkunlp/instructor-base"
_BATCH_SIZE = 8     # safe for CPU; bump on GPU if you have one


class Embedder:
    def __init__(self, model_id: str = _MODEL_ID, device: str = None):
        self.model_id = model_id
        self.device = device  # None lets sentence-transformers auto-pick
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            print(f"[embedder] loading {self.model_id}...", flush=True)
            self._model = SentenceTransformer(
                self.model_id, device=self.device, trust_remote_code=True
            )
            print(f"[embedder] ready (device={self._model.device})", flush=True)

    # ---- public API --------------------------------------------------

    def embed_chunks(self, texts: Iterable[str]) -> np.ndarray:
        """
        Embed a batch of course-content chunks for indexing.
        Returns an (N, 768) numpy array of float32.
        """
        self._ensure_model()
        texts = list(texts)
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        # INSTRUCTOR expects [[instruction, text], [instruction, text], ...]
        pairs = [[INDEX_INSTRUCTION, t] for t in texts]
        embs = self._model.encode(
            pairs,
            batch_size=_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # so cosine == dot product
        )
        return embs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single student question. Returns a (768,) numpy array.
        Uses the query instruction to bias the embedding toward 'what is
        this question really asking?' rather than 'what does this say?'
        """
        self._ensure_model()
        pair = [[QUERY_INSTRUCTION, query]]
        emb = self._model.encode(
            pair,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb[0].astype(np.float32)
