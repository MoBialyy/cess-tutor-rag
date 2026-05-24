"""
Maximal Marginal Relevance (MMR) diversification.

Per report section 5.4.7, after the cross-encoder reranks to top-N,
MMR selects a final top-K subset that balances:
  - relevance to the query
  - diversity among the selected chunks

MMR formula:
  MMR(d) = lambda * sim(d, q) - (1 - lambda) * max_j sim(d, d_j)
where d_j ranges over already-selected chunks.

We work directly with the cross-encoder rerank scores as "relevance to
query", and use INSTRUCTOR embeddings of the chunks themselves for
"similarity to selected chunks". The embeddings are already in the
candidates (passed in from the dense retriever).
"""

from typing import List, Dict
import numpy as np


def mmr_select(
    candidates: List[Dict],
    top_k: int = 5,
    lambda_param: float = 0.6,
) -> List[Dict]:
    """
    Pick top_k candidates using MMR.

    Each candidate dict must have:
      - 'rerank_score': float, relevance to the query (higher = better)
      - 'embedding':    np.ndarray, the chunk's INSTRUCTOR embedding
                        (used for inter-chunk similarity)

    lambda_param controls the trade-off:
      1.0 = pure relevance (no diversity penalty)
      0.0 = pure diversity (no relevance signal)
      0.6 = report-aligned default (slight bias toward relevance)
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    # Normalize rerank scores to [0, 1] so they're comparable to cosine
    # similarities between embeddings (which are also in roughly [0, 1]
    # since embeddings are normalized).
    scores = np.array([c["rerank_score"] for c in candidates], dtype=np.float32)
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-6:
        rel = (scores - s_min) / (s_max - s_min)
    else:
        rel = np.ones_like(scores)

    embeddings = np.stack([c["embedding"] for c in candidates])  # (N, dim)

    selected_idx: List[int] = []
    remaining_idx = list(range(len(candidates)))

    # First pick: highest relevance
    first = int(np.argmax(rel))
    selected_idx.append(first)
    remaining_idx.remove(first)

    while len(selected_idx) < top_k and remaining_idx:
        # For each remaining candidate, compute max similarity to anything
        # already selected.
        sel_embs = embeddings[selected_idx]      # (k_sel, dim)
        best_idx = None
        best_score = -np.inf

        for i in remaining_idx:
            sim_to_selected = float(np.max(sel_embs @ embeddings[i]))
            mmr_score = (
                lambda_param * rel[i]
                - (1.0 - lambda_param) * sim_to_selected
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [candidates[i] for i in selected_idx]
