from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from .crawler import Page


def is_article_url(url: str) -> bool:
    """
    Determine if a URL represents an actual article content page.
    Excludes author pages, tag pages, category pages, etc.
    """
    # Must contain /articles path
    if "/articles" not in url:
        return False
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url.lower()
    
    # Exclude non-article pages
    exclude_patterns = [
        "/author/", "/authors/",
        "/tag/", "/tags/", 
        "/category/", "/categories/",
        "/page/",
        "/archive",
        "/guides/",  # Guide listings
        "/about/", "/contact/", "/privacy/", "/terms/"
    ]
    
    for pattern in exclude_patterns:
        if pattern in url_lower:
            return False
    
    # Exclude root articles listing page
    if url_lower.rstrip('/').endswith('/articles'):
        return False
    
    # If it passes all filters and contains /articles, it's likely an article
    return True


def build_link_graph(pages: Dict[str, Page]) -> nx.DiGraph:
    G = nx.DiGraph()

    for url, page in pages.items():
        G.add_node(
            url,
            title=page.title,
            is_article=is_article_url(url),
            depth=page.depth,
        )

    edge_anchor_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for src, page in pages.items():
        for link in page.links:
            if link.href in pages:  # only keep edges within crawled set
                edge_anchor_map[(src, link.href)].append(link.anchor)

    for (src, dst), anchors in edge_anchor_map.items():
        G.add_edge(src, dst, weight=len(anchors), anchors=anchors[:50])

    return G


def export_graph_csv(G: nx.DiGraph, nodes_csv_path: str, edges_csv_path: str) -> None:
    import pandas as pd

    nodes_records = []
    for node, data in G.nodes(data=True):
        nodes_records.append({"url": node, **data})

    edges_records = []
    for src, dst, data in G.edges(data=True):
        anchors = data.get("anchors", [])
        edges_records.append(
            {
                "source": src,
                "target": dst,
                "weight": data.get("weight", 1),
                "anchors": " | ".join(sorted(set(a for a in anchors if a)))[:2000],
            }
        )

    pd.DataFrame(nodes_records).to_csv(nodes_csv_path, index=False)
    pd.DataFrame(edges_records).to_csv(edges_csv_path, index=False)


def compute_centrality_metrics(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    bc = nx.betweenness_centrality(G, k=min(500, max(10, G.number_of_nodes()))) if G.number_of_nodes() > 100 else nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)
    metrics: Dict[str, Dict[str, float]] = {}
    for n in G.nodes():
        metrics[n] = {
            "pagerank": float(pr.get(n, 0.0)),
            "betweenness": float(bc.get(n, 0.0)),
            "closeness": float(cc.get(n, 0.0)),
            "in_degree": float(G.in_degree(n)),
            "out_degree": float(G.out_degree(n)),
        }
    return metrics


def filter_article_only(G: nx.DiGraph) -> nx.DiGraph:
    H = G.copy()
    remove_nodes = [n for n, d in H.nodes(data=True) if not d.get("is_article")]
    H.remove_nodes_from(remove_nodes)
    return H
