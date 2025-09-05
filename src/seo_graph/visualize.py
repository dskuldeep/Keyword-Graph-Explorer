from __future__ import annotations

from typing import Optional

import networkx as nx
from pyvis.network import Network


def visualize_pyvis(
    G: nx.DiGraph,
    output_html_path: str,
    height: str = "900px",
    width: str = "100%",
    show_buttons: bool = True,
) -> None:
    net = Network(height=height, width=width, directed=True, notebook=False, cdn_resources="in_line")
    net.force_atlas_2based()

    for node, data in G.nodes(data=True):
        title = data.get("title", "")
        cluster = data.get("cluster")
        sim = data.get("pillar_sim")
        tooltip_parts = [f"<b>Title</b>: {title}"] if title else []
        if cluster is not None:
            tooltip_parts.append(f"<b>Cluster</b>: {cluster}")
        if sim is not None:
            tooltip_parts.append(f"<b>Pillar sim</b>: {sim:.3f}")
        tooltip = "<br/>".join(tooltip_parts)

        net.add_node(
            node,
            label=(title[:40] + "…") if title and len(title) > 40 else (title or node),
            title=tooltip or node,
            color="#1f78b4" if data.get("is_article") else "#a6cee3",
        )

    for src, dst, data in G.edges(data=True):
        anchors = data.get("anchors", [])
        unique_anchors = sorted(set(a for a in anchors if a))
        title = "<br/>".join(unique_anchors[:20])
        net.add_edge(src, dst, value=data.get("weight", 1), title=title)

    if show_buttons:
        net.show_buttons(filter_=["physics"])
    # Write HTML directly to avoid environment-specific template issues in pyvis.show
    net.write_html(output_html_path, open_browser=False, notebook=False)
