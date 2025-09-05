from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Recommendation:
    source_url: str
    target_url: str
    similarity: float
    reason: str


def recommend_interlinks(
    article_urls: List[str],
    texts: List[str],
    embeddings: np.ndarray,
    G: nx.DiGraph,
    top_n: int = 3,
    min_sim: float = 0.5,
) -> List[Recommendation]:
    if len(article_urls) != embeddings.shape[0]:
        raise ValueError("article_urls and embeddings length mismatch")

    sims = cosine_similarity(embeddings)
    recs: List[Recommendation] = []

    for i, u in enumerate(article_urls):
        # Exclude self and already linked targets
        existing = set(G.successors(u))
        scores = []
        for j, v in enumerate(article_urls):
            if i == j or v in existing:
                continue
            score = float(sims[i, j])
            if score >= min_sim:
                scores.append((v, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        for v, score in scores[:top_n]:
            reason = "High semantic similarity; add contextual link"
            recs.append(Recommendation(source_url=u, target_url=v, similarity=score, reason=reason))

    return recs


def export_recommendations_csv(path: str, recs: List[Recommendation]) -> None:
    import pandas as pd

    rows = [
        {
            "source_url": r.source_url,
            "target_url": r.target_url,
            "similarity": round(r.similarity, 4),
            "reason": r.reason,
        }
        for r in recs
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
