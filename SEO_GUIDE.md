# Using the SEO Graph Explorer for Internal Linking & Topic Clusters

This guide explains how to use the outputs and dashboard to improve SEO.

## Files produced (in `output/<name>`)
- `graph.html`: Interactive internal link graph
- `nodes.csv` / `edges.csv`: Full graph nodes/edges with attributes and anchor texts
- `nodes_articles.csv` / `edges_articles.csv`: Article-only subset
- `centrality.csv`: PageRank, betweenness, closeness, in/out degree for articles
- `keywords.csv`: Per-article TF‑IDF keywords
- `recommendations.csv`: Suggested article interlinks based on semantic similarity
- `crawl.json`: Raw crawl with titles, text, anchors

## Dashboard (Streamlit)
- Cluster Overview: counts per cluster; use to spot thin clusters
- Top Central Articles: use to identify pillar/hub pages
- Keyword Explorer: see per-article TF‑IDF terms for on-page optimization
- Interlink Recommendations: quick wins for internal links
- Anchor Text Search: find weak/irrelevant anchors to fix
- Interactive Network Graph: filter by clusters and article-only; inspect edges and anchors

## How to use for SEO
1. Ensure coverage: run with sitemap and focus on `/articles` to include all articles.
2. Identify pillars and hubs: sort `centrality.csv` by `pagerank` to find candidates.
3. Strengthen clusters: for each cluster, link related articles (use `recommendations.csv`).
4. Fix orphan/low-degree pages: in `centrality.csv`, filter low in/out degree and add links to/from relevant hubs.
5. Improve anchor texts: search weak anchors (e.g., "click here") in the dashboard; rewrite to keyword-rich variants.
6. Optimize on-page terms: use `keywords.csv` to align titles/H1/intro with top TF‑IDF terms; add missing terms naturally.
7. Balance crosslinks: avoid over-linking across clusters unless clearly relevant; prioritize within-cluster linking.
8. Monitor pillar relevance: use `pillar_sim` in `nodes.csv` to find low-sim articles; revise content or links.
9. Build topic pages: clusters with many articles and strong terms are good candidates for pillar or hub pages.

## Re-running
Use the CLI to regenerate:
```
python -m seo_graph.cli all \
  --seed https://getmaxim.ai/articles \
  --domain getmaxim.ai \
  --sitemap https://getmaxim.ai/articles/sitemap.xml \
  --focus-prefix https://getmaxim.ai/articles \
  --out output/getmaxim \
  --max-pages 800 --max-depth 3 \
  --pillar "AI marketing analytics platform"
```

## Tips
- If you change content or site structure, re-run and compare `centrality.csv` and `cluster` shifts.
- For very large sites, raise `--max-pages` and use `--focus-prefix` to constrain expansion.
- Export filtered views from the dashboard for handoff to content/SEO teams.
