from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dataclasses import asdict

from .crawler import crawl_site
from .graph_builder import build_link_graph, export_graph_csv, compute_centrality_metrics, filter_article_only
from .nlp_cluster import (
    embed_documents,
    cluster_documents,
    compute_pillar_similarity,
    extract_keywords_per_doc,
    label_clusters_by_keywords,
    compute_hybrid_embeddings,
    compute_umap_2d,
)
from .visualize import visualize_pyvis
from .report import recommend_interlinks, export_recommendations_csv


def run_all(seed: str, domain: str, out_dir: str, max_pages: int, max_depth: int, pillar: str | None,
            sitemap: str | None = None, focus_prefix: str | None = None, clusters: int | None = None,
            doc_keywords_topk: int = 20, cluster_keywords_topk: int = 15) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pages = crawl_site(
        seed,
        allowed_domain=domain,
        max_pages=max_pages,
        max_depth=max_depth,
        sitemap_url=sitemap,
        focus_prefix=focus_prefix,
    )

    crawl_path = out / "crawl.json"
    with crawl_path.open("w") as f:
        json.dump({k: asdict(v) for k, v in pages.items()}, f)

    G = build_link_graph(pages)

    # Embeddings for article pages only
    article_nodes = [n for n, d in G.nodes(data=True) if d.get("is_article")]
    texts = [pages[n].text or pages[n].title for n in article_nodes]
    if len(texts) >= 2:
        # Hybrid embeddings (SBERT + TF-IDF SVD) for clustering/UMAP
        embeddings = compute_hybrid_embeddings(texts)
        labels = cluster_documents(embeddings, k=clusters or 7)
        # Use SBERT-only embeddings for pillar similarity to avoid dimension mismatch
        sbert_embeddings = embed_documents(texts)
        sim = compute_pillar_similarity(sbert_embeddings, pillar)
        for i, n in enumerate(article_nodes):
            G.nodes[n]["cluster"] = int(labels[i])
            if sim is not None:
                G.nodes[n]["pillar_sim"] = float(sim[i])

        # Generate link recommendations between similar articles
        recs = recommend_interlinks(article_nodes, texts, embeddings, G, top_n=3, min_sim=0.5)
        export_recommendations_csv(str(out / "recommendations.csv"), recs)

        # Keyword extraction and cluster labels
        per_doc_keywords, terms = extract_keywords_per_doc(texts, top_k=int(doc_keywords_topk))
        cluster_labels = label_clusters_by_keywords(texts, labels, terms, per_doc_keywords, top_k=int(cluster_keywords_topk))
        # Export per-article keywords
        kw_out = out / "keywords.csv"
        import pandas as pd
        pd.DataFrame(
            {
                "url": article_nodes,
                "keywords": ["; ".join(k for k, _ in ks) for ks in per_doc_keywords],
            }
        ).to_csv(kw_out, index=False)
        # Attach cluster labels to nodes
        for lab, label_terms in cluster_labels.items():
            for node, y in zip(article_nodes, labels):
                if int(y) == int(lab):
                    labels_str = ", ".join(label_terms)
                    G.nodes[node]["cluster_label"] = labels_str
                    G.nodes[node]["cluster_keywords"] = labels_str

        # Pillar similarity per article based on cluster keyword centroid (TF-IDF space)
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True, stop_words="english", min_df=2)
        tfidf_mat = tfidf_vec.fit_transform(texts)
        # Compute normalized centroid per cluster
        cluster_ids = sorted(set(int(c) for c in set(labels.tolist())))
        cluster_to_idxs = {c: [i for i, _ in enumerate(article_nodes) if int(labels[i]) == c] for c in cluster_ids}
        centroid = {}
        for c, idxs in cluster_to_idxs.items():
            if not idxs:
                centroid[c] = None
                continue
            mean = tfidf_mat[idxs].mean(axis=0)
            vec = np.asarray(mean).ravel().astype(np.float32)
            nrm = np.linalg.norm(vec) + 1e-9
            centroid[c] = vec / nrm
        # Assign similarity to each article node
        for i, n in enumerate(article_nodes):
            c = int(labels[i])
            doc = tfidf_mat.getrow(i)
            cl = centroid.get(c)
            if cl is None or doc.nnz == 0:
                sim_val = 0.0
            else:
                sim_val = float(np.asarray(doc.dot(cl)).ravel()[0])
            G.nodes[n]["pillar_sim"] = sim_val

        # Save embeddings for dynamic clustering in the app
        import numpy as np
        np.save(out / "embeddings_hybrid.npy", embeddings)
        np.save(out / "embeddings_sbert.npy", sbert_embeddings)

        # UMAP 2D coordinates for visualization
        coords = compute_umap_2d(embeddings)
        umap_out = out / "umap.csv"
        pd.DataFrame({
            "url": article_nodes,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": labels,
        }).to_csv(umap_out, index=False)

        # Cluster keywords export
        pd.DataFrame(
            [{"cluster": int(c), "keywords": ", ".join(ks)} for c, ks in cluster_labels.items()]
        ).to_csv(out / "cluster_keywords.csv", index=False)

        # Export per-article keywords (explicit top N)
        pd.DataFrame(
            {
                "url": article_nodes,
                "title": [G.nodes[n].get("title", "") for n in article_nodes],
                "cluster": labels,
                "keywords_top": ["; ".join(k for k, _ in ks) for ks in per_doc_keywords],
            }
        ).to_csv(out / "article_keywords.csv", index=False)

    # Metrics and exports
    export_graph_csv(G, str(out / "nodes.csv"), str(out / "edges.csv"))

    # Article-only graph and centrality
    H = filter_article_only(G)
    metrics = compute_centrality_metrics(H)
    import pandas as pd
    pd.DataFrame(
        [{"url": u, **m} for u, m in metrics.items()]
    ).to_csv(out / "centrality.csv", index=False)
    export_graph_csv(H, str(out / "nodes_articles.csv"), str(out / "edges_articles.csv"))

    # Identify pillars per cluster (top PageRank within cluster) and compute similarities
    if len(texts) >= 2:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Build helper maps
        url_to_cluster = {n: G.nodes[n].get("cluster") for n in article_nodes}
        pr = {u: metrics.get(u, {}).get("pagerank", 0.0) for u in article_nodes}

        # Compute SBERT centroid per cluster
        cluster_ids = sorted(set(int(c) for c in set(labels.tolist())))
        cluster_to_indices: dict[int, list[int]] = {c: [i for i, u in enumerate(article_nodes) if int(labels[i]) == c] for c in cluster_ids}
        sbert_norm = sbert_embeddings / (np.linalg.norm(sbert_embeddings, axis=1, keepdims=True) + 1e-9)
        cluster_sbert_centroid: dict[int, np.ndarray] = {
            c: np.mean(sbert_norm[idxs], axis=0) if idxs else np.zeros(sbert_norm.shape[1], dtype=np.float32)
            for c, idxs in cluster_to_indices.items()
        }
        for c, vec in cluster_sbert_centroid.items():
            nrm = np.linalg.norm(vec) + 1e-9
            cluster_sbert_centroid[c] = (vec / nrm).astype(np.float32)

        # TF-IDF matrix and centroid per cluster
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True, stop_words="english", min_df=2)
        tfidf = vectorizer.fit_transform(texts)
        cluster_tfidf_centroid: dict[int, np.ndarray] = {}
        for c, idxs in cluster_to_indices.items():
            if not idxs:
                cluster_tfidf_centroid[c] = np.zeros((tfidf.shape[1],), dtype=np.float32)
            else:
                centroid = tfidf[idxs].mean(axis=0)
                centroid = np.asarray(centroid).ravel().astype(np.float32)
                norm = np.linalg.norm(centroid) + 1e-9
                cluster_tfidf_centroid[c] = centroid / norm

        # Build DataFrame for pillar selection
        df_nodes = pd.DataFrame({
            "url": article_nodes,
            "cluster": labels,
            "title": [G.nodes[n].get("title", "") for n in article_nodes],
            "pagerank": [pr.get(u, 0.0) for u in article_nodes],
            "text_len": [len((pages[u].text or "").split()) for u in article_nodes],
        })

        pillar_rows = []
        for c, grp in df_nodes.groupby("cluster"):
            if grp.empty:
                continue
            # Exclude listing/author/tag/archive pages and very short texts
            cand = grp.copy()
            # content length requirement
            cand = cand[cand["text_len"] >= 150]
            # robust URL filtering (lowercased) - strengthen author page filtering
            low = cand["url"].str.lower()
            mask_bad = (
                low.str.contains(r"/author/|/authors/|/tag/|/tags/|/categories/|/category/|/archive|/guides", regex=True)
                | low.str.contains(r"/page/", regex=True)
                | low.str.fullmatch(r"https?://[^/]+/articles/?")
                | low.str.contains(r"/articles/page/", regex=True)
                | low.str.contains(r"/articles/author/", regex=True)  # Explicit author page filtering
                | low.str.contains(r"/about/|/contact/|/privacy/|/terms/", regex=True)  # Additional non-content pages
            )
            cand = cand[~mask_bad]
            if cand.empty:
                cand = grp

            scores = []
            for _, row in cand.iterrows():
                u = row["url"]
                i = article_nodes.index(u)
                sem_sim = float(np.dot(sbert_norm[i], cluster_sbert_centroid[int(c)]))
                doc_vec = tfidf.getrow(i)
                cl_vec = cluster_tfidf_centroid[int(c)]
                if doc_vec.nnz > 0:
                    dot_val = doc_vec.dot(cl_vec)
                    import numpy as np  # safe local import
                    tfidf_sim = float(np.asarray(dot_val).ravel()[0])
                else:
                    tfidf_sim = 0.0
                total_sim = 0.6 * sem_sim + 0.4 * tfidf_sim
                scores.append((total_sim, row["pagerank"], u, row["title"], sem_sim, tfidf_sim))
            scores.sort(reverse=True)
            total_sim, pr_best, u, title_u, sem_sim, tfidf_sim = scores[0]
            excerpt = (pages[u].text or pages[u].title or "")[:500]
            pillar_rows.append({
                "cluster": int(c),
                "url": u,
                "title": title_u,
                "pagerank": float(pr_best),
                "semantic_sim": round(sem_sim, 4),
                "tfidf_sim": round(tfidf_sim, 4),
                "total_sim": round(total_sim, 4),
                "cluster_keywords": ", ".join(cluster_labels.get(int(c), [])),
                "excerpt": excerpt,
            })

        pd.DataFrame(pillar_rows).sort_values(["cluster"], ascending=True).to_csv(out / "pillars.csv", index=False)

    visualize_pyvis(G, str(out / "graph.html"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SEO Graph: crawl and visualize internal linking with clustering")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("all", help="Run crawl + graph + cluster + visualize")
    p_all.add_argument("--seed", required=True, help="Seed URL, e.g., https://getmaxim.ai/articles")
    p_all.add_argument("--domain", required=True, help="Allowed domain, e.g., getmaxim.ai")
    p_all.add_argument("--out", required=True, help="Output directory")
    p_all.add_argument("--max-pages", type=int, default=400, help="Max pages to crawl")
    p_all.add_argument("--max-depth", type=int, default=3, help="Max crawl depth")
    p_all.add_argument("--pillar", default=None, help="Pillar topic text to compute similarity")
    p_all.add_argument("--sitemap", default=None, help="Sitemap URL to seed crawl (e.g., https://.../sitemap.xml)")
    p_all.add_argument("--focus-prefix", default=None, help="Only expand URLs under this prefix (e.g., https://getmaxim.ai/articles)")
    p_all.add_argument("--clusters", type=int, default=7, help="Number of clusters for topic grouping")
    p_all.add_argument("--doc-keywords-topk", type=int, default=20, help="Top N keywords per article")
    p_all.add_argument("--cluster-keywords-topk", type=int, default=15, help="Top N keywords per cluster")

    args = parser.parse_args()

    if args.cmd == "all":
        run_all(
            args.seed,
            args.domain,
            args.out,
            args.max_pages,
            args.max_depth,
            args.pillar,
            args.sitemap,
            args.focus_prefix,
            args.clusters,
            args.doc_keywords_topk,
            args.cluster_keywords_topk,
        )


if __name__ == "__main__":
    main()
