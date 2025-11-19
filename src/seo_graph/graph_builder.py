from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from .crawler import Page


def is_article_url(url: str, config: dict = None, allowed_domain: str = None) -> bool:
    """
    Determine if a URL represents an actual article content page.
    Excludes author pages, tag pages, category pages, etc.
    """
    from urllib.parse import urlparse
    import tldextract
    
    # Load config if not provided
    if config is None:
        config = load_article_detection_config()
    
    # Check if this blog should skip path pattern checking (subdomain-based blogs)
    skip_path_check = False
    if allowed_domain:
        try:
            from pathlib import Path
            import json
            config_path = Path("blog_config.json")
            if config_path.exists():
                with config_path.open() as f:
                    blog_config = json.load(f)
                blog_paths = blog_config.get("blog_paths", [])
                for blog in blog_paths:
                    if blog.get("domain", "").rstrip('/') == allowed_domain.rstrip('/'):
                        skip_path_check = blog.get("skip_path_pattern_check", False)
                        break
        except Exception:
            pass
    
    url_patterns = config.get("url_patterns", ["/articles/", "/blog/"])
    exclude_patterns = config.get("exclude_patterns", [
        "/author/", "/authors/",
        "/tag/", "/tags/", 
        "/category/", "/categories/",
        "/page/",
        "/archive",
        "/guides/",  # Guide listings
        "/about/", "/contact/", "/privacy/", "/terms/"
    ])
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url.lower()
    
    # If skip_path_check is enabled, treat all URLs on the domain as articles unless excluded
    if skip_path_check and allowed_domain:
        url_netloc = urlparse(url).netloc
        url_extracted = tldextract.extract(url_netloc)
        allowed_extracted = tldextract.extract(allowed_domain.rstrip('/'))
        
        # Check if URL is on the same domain
        if (url_extracted.domain == allowed_extracted.domain and 
            url_extracted.suffix == allowed_extracted.suffix and
            url_extracted.subdomain == allowed_extracted.subdomain):
            # Exclude non-article pages
            for pattern in exclude_patterns:
                if pattern in url_lower:
                    return False
            # Exclude root/homepage
            url_path = urlparse(url).path.rstrip('/')
            if not url_path or url_path == '/':
                return False
            # If it passes filters, it's an article
            return True
    
    # Original logic for path-based blogs
    # Check if URL matches any of the article patterns
    matches_article_pattern = any(pattern in url_lower for pattern in url_patterns)
    if not matches_article_pattern:
        return False
    
    # Exclude non-article pages
    for pattern in exclude_patterns:
        if pattern in url_lower:
            return False
    
    # Exclude root listing pages (e.g., /articles, /blog)
    for pattern in url_patterns:
        if url_lower.rstrip('/').endswith(pattern.rstrip('/')):
            return False
    
    # If it passes all filters and matches article patterns, it's likely an article
    return True


def load_article_detection_config() -> dict:
    """Load article detection configuration from blog_config.json"""
    try:
        from pathlib import Path
        import json
        
        config_path = Path("blog_config.json")
        if config_path.exists():
            with config_path.open() as f:
                config = json.load(f)
            return config.get("article_detection", {})
    except Exception:
        pass
    
    # Return default config
    return {
        "url_patterns": ["/articles/", "/blog/", "/posts/", "/news/", "/insights/", "/updates/"],
        "exclude_patterns": [
            "/author/", "/authors/", "/tag/", "/tags/", "/category/", "/categories/",
            "/page/", "/archive", "/guides/", "/about/", "/contact/", "/privacy/", "/terms/",
            "/search", "/feed", "/rss", "/sitemap"
        ],
        "title_exclude_patterns": ["blog", "author", "tag", "category", "search", "page"],
        "min_title_length": 10
    }


def build_link_graph(pages: Dict[str, Page], allowed_domain: str = None) -> nx.DiGraph:
    G = nx.DiGraph()

    for url, page in pages.items():
        G.add_node(
            url,
            title=page.title,
            is_article=is_article_url(url, allowed_domain=allowed_domain),
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
