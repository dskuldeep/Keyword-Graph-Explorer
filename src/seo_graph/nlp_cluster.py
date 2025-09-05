from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embed_documents(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def choose_k(num_items: int) -> int:
    if num_items <= 5:
        return 1
    if num_items <= 20:
        return 2
    if num_items <= 60:
        return 3
    return max(4, int(np.sqrt(num_items)))


def cluster_documents(embeddings: np.ndarray, k: Optional[int] = None, random_state: int = 42) -> np.ndarray:
    n = embeddings.shape[0]
    k = k or choose_k(n)
    k = max(1, min(k, n))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(embeddings)
    return labels


def compute_pillar_similarity(
    doc_embeddings: np.ndarray,
    pillar_text: Optional[str],
    model_name: str = DEFAULT_MODEL,
) -> Optional[np.ndarray]:
    if pillar_text is None:
        return None
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    model = SentenceTransformer(model_name)
    pillar_vec = model.encode([pillar_text], convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(doc_embeddings, pillar_vec)[..., 0]
    return sims.astype(np.float32)


def extract_keywords_per_doc(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    top_k: int = 15,
) -> Tuple[List[List[Tuple[str, float]]], List[str]]:
    """Return per-document top keywords and the vocabulary terms.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words="english",
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out().tolist()

    per_doc: List[List[Tuple[str, float]]] = []
    for i in range(tfidf.shape[0]):
        row = tfidf.getrow(i)
        items = list(zip(row.indices, row.data))
        items.sort(key=lambda x: x[1], reverse=True)
        top = [(terms[idx], float(score)) for idx, score in items[:top_k]]
        per_doc.append(top)
    return per_doc, terms


def label_clusters_by_keywords(
    texts: List[str],
    labels: np.ndarray,
    terms: List[str],
    tfidf_top_per_doc: List[List[Tuple[str, float]]],
    top_k: int = 7,
) -> Dict[int, List[str]]:
    """Aggregate per-doc keywords to produce cluster labels.
    """
    cluster_to_terms: Dict[int, Dict[str, float]] = {}
    for doc_idx, lab in enumerate(labels):
        store = cluster_to_terms.setdefault(int(lab), {})
        for term, score in tfidf_top_per_doc[doc_idx]:
            store[term] = store.get(term, 0.0) + score

    cluster_labels: Dict[int, List[str]] = {}
    for lab, term_scores in cluster_to_terms.items():
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        cluster_labels[lab] = [t for t, _ in sorted_terms[:top_k]]
    return cluster_labels


def compute_hybrid_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    tfidf_max_features: int = 10000,
    svd_components: int = 64,
    alpha: float = 0.5,
) -> np.ndarray:
    """Concatenate normalized SBERT embeddings with TF-IDF SVD components.

    alpha blends contributions: 1.0 => only SBERT, 0.0 => only TF-IDF SVD.
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(svd_components, max(2, min(tfidf.shape) - 1)), random_state=42)
    tfidf_svd = svd.fit_transform(tfidf)

    # L2 normalize components
    def l2norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / n

    emb = l2norm(emb)
    tfidf_svd = l2norm(tfidf_svd)

    a = max(0.0, min(1.0, float(alpha)))
    emb_scaled = emb * np.sqrt(a)
    tfidf_scaled = tfidf_svd * np.sqrt(1.0 - a)
    hybrid = np.concatenate([emb_scaled, tfidf_scaled], axis=1).astype(np.float32)
    return hybrid


def compute_umap_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    try:
        import umap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("umap-learn not available") from exc
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    coords = reducer.fit_transform(embeddings)
    return coords.astype(np.float32)
