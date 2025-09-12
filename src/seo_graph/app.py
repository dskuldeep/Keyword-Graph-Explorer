import streamlit as st
import pandas as pd
import json
from pathlib import Path
import math
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import altair as alt
from scipy.spatial import ConvexHull
import numpy as np

st.set_page_config(page_title="SEO Graph Explorer", layout="wide", initial_sidebar_state="expanded")

# Multi-path configuration (defined early for use in header)
def load_blog_config():
    """Load blog paths from config file."""
    config_path = Path("blog_config.json")
    if config_path.exists():
        try:
            with config_path.open() as f:
                config = json.load(f)
            blog_paths = config.get("blog_paths", [])
            return {blog["name"]: blog["output_dir"] for blog in blog_paths}
        except Exception as e:
            st.error(f"Error loading blog config: {e}")
    
    # Fallback to default paths
    return {
        "getmaxim.ai/articles": "getmaxim",
        "getmaxim.ai/blog": "getmaxim_blog"
    }

available_paths = load_blog_config()
selected_site = list(available_paths.keys())[0] if available_paths else "getmaxim.ai/articles"  # Default value, will be updated by sidebar

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Dynamic header with current site context
st.markdown(f'''
<div class="main-header">
    <h1>🔍 SEO Graph Explorer</h1>
    <p>Advanced Content Clustering & Link Analysis for SEO & GEO Optimization</p>
    <p style="font-size: 1.1em; margin-top: 1rem; opacity: 0.9;">
        📍 Currently analyzing: <strong>{selected_site}</strong>
    </p>
</div>
''', unsafe_allow_html=True)

# Multi-path information
with st.expander("🌐 Multi-Path Analysis", expanded=False):
    st.markdown(f"""
    ### **Current Analysis: {selected_site}**
    
    **📊 Data Status:**
    - **Data Directory:** `output/{available_paths[selected_site]}`
    - **Analysis Type:** Independent content analysis
    - **Data Isolation:** Each path maintains separate analysis data
    
    **🔄 Available Paths:**
    {chr(10).join([f"    - **{name}** → `output/{output_dir}/`" for name, output_dir in available_paths.items()])}
    
    **💡 How It Works:**
    - Each path is analyzed independently with its own clustering and metrics
    - Switch between paths using the sidebar dropdown
    - All analysis data (clusters, keywords, recommendations) is path-specific
    - Perfect for comparing different sections of your site or multiple sites
    """)

# Workflow Overview
with st.expander("🚀 How This System Works - Complete Workflow", expanded=False):
    st.markdown("""
    ### 🔄 **End-to-End SEO Analysis Pipeline**
    
    **1. 🕷️ Content Discovery & Crawling**
    - Discovers all pages via sitemap parsing and recursive crawling
    - Extracts clean content using advanced NLP (trafilatura)
    - Maps internal link structure with anchor text analysis
    
    **2. 🧠 AI-Powered Content Understanding**
    - **Semantic Embeddings**: Uses Sentence-BERT to understand content meaning
    - **Keyword Extraction**: TF-IDF analysis identifies important terms
    - **Hybrid Intelligence**: Combines semantic + keyword signals for robust clustering
    
    **3. 📊 Topic Clustering & Analysis**
    - Groups content by semantic similarity and keyword patterns
    - Identifies content gaps and opportunities
    - Labels clusters with representative keywords
    
    **4. 🔗 Link Authority & Structure Analysis**
    - Computes PageRank for content authority scoring
    - Identifies bridge content (betweenness centrality)
    - Maps content connectivity and influence
    
    **5. 🎯 Pillar Content Identification**
    - Automatically identifies the most representative content per topic
    - Scores based on semantic relevance to cluster themes
    - Helps structure your content hierarchy
    
    **6. 💡 Actionable SEO Recommendations**
    - Suggests internal linking opportunities
    - Identifies content gaps in your topic coverage
    - Provides anchor text optimization insights
    """)

# System Benefits for SEO/GEO
with st.expander("🎯 SEO & Generative Engine Optimization Benefits", expanded=False):
    st.markdown("""
    ### **For Traditional SEO:**
    - **Content Gap Analysis**: Identify missing topics in your content strategy
    - **Internal Linking Strategy**: Data-driven link building recommendations
    - **Topic Authority**: Build comprehensive topic clusters for expertise signals
    - **Anchor Text Optimization**: Understand and improve your internal anchor text strategy
    
    ### **For Generative Engine Optimization (GEO):**
    - **Semantic Coherence**: Ensure your content forms logical, AI-readable topic clusters
    - **Knowledge Graph Building**: Create interconnected content that AI can easily understand
    - **Context Richness**: Identify relationships between concepts for better AI comprehension
    - **Authority Signals**: Build clear content hierarchies that AI systems recognize
    """)

st.sidebar.title("📋 Navigation")
st.sidebar.markdown("Use this panel to navigate through different analysis sections.")

# Data Loading Section
st.sidebar.subheader("📁 Data Configuration")

# Multi-path support with dropdown selector
selected_site = st.sidebar.selectbox(
    "🌐 Select Site/Path", 
    options=list(available_paths.keys()),
    index=0,
    help="Choose which site's data to analyze. Each path maintains separate analysis data."
)

# Display current path context
st.sidebar.markdown(f"**📍 Current Path:** `{selected_site}`")
st.sidebar.markdown(f"**📂 Data Directory:** `output/{available_paths[selected_site]}`")

# Set the output directory based on selected path
out_dir = str(Path.cwd() / "output" / available_paths[selected_site])
base = Path(out_dir)

nodes_file = base / "nodes.csv"
edges_file = base / "edges.csv"
article_nodes_file = base / "nodes_articles.csv"
article_edges_file = base / "edges_articles.csv"
centrality_file = base / "centrality.csv"
keywords_file = base / "keywords.csv"
recs_file = base / "recommendations.csv"
umap_file = base / "umap.csv"
cluster_kw_file = base / "cluster_keywords.csv"
pillars_file = base / "pillars.csv"
article_kws_file = base / "article_keywords.csv"

# Key Metrics Dashboard
st.markdown(f"## 📊 **Site Analysis Overview - {selected_site}**")

cols = st.columns(4)
with cols[0]:
    if nodes_file.exists():
        nodes = pd.read_csv(nodes_file)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌐 Total Pages", len(nodes))
        st.markdown("*All discovered pages*")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"nodes.csv not found for {selected_site}")
        st.info(f"💡 **To analyze {selected_site}:** Run the CLI tool with this path as the seed URL")
        nodes = pd.DataFrame()

with cols[1]:
    if edges_file.exists():
        edges = pd.read_csv(edges_file)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔗 Internal Links", len(edges))
        st.markdown("*Link connections found*")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"edges.csv not found for {selected_site}")
        edges = pd.DataFrame()

with cols[2]:
    if centrality_file.exists() and centrality_file.stat().st_size > 1:
        try:
            centrality = pd.read_csv(centrality_file)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📝 Article Pages", len(centrality))
            st.markdown("*Content for analysis*")
            st.markdown('</div>', unsafe_allow_html=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            st.warning(f"centrality.csv is empty for {selected_site}")
            centrality = pd.DataFrame()
    else:
        st.warning(f"centrality.csv not found or empty for {selected_site}")
        centrality = pd.DataFrame()

with cols[3]:
    if not nodes.empty:
        clusters = 0
        if "cluster" in nodes.columns and not nodes[nodes["is_article"] == True]["cluster"].dropna().empty:
            clusters = len(nodes[nodes["is_article"] == True]["cluster"].dropna().unique())
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Topic Clusters", clusters)
        st.markdown("*AI-identified themes*")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"## 🎯 **Topic Clusters Analysis - {selected_site}**")

st.markdown("""
<div class="info-box">
<strong>💡 SEO Insight:</strong> Topic clusters help you understand how your content is organized thematically. 
Well-clustered content signals topical authority to search engines and AI systems.
</div>
""", unsafe_allow_html=True)

if not nodes.empty:
    df = nodes[nodes["is_article"] == True].copy()
    if "cluster_keywords" not in df.columns and "cluster_label" in df.columns:
        df["cluster_keywords"] = df["cluster_label"]
    
    # Check if cluster column exists and has data
    if "cluster" in df.columns and not df["cluster"].dropna().empty:
        cluster_counts = df.groupby("cluster").agg(
            count=("url", "size"),
            keywords=("cluster_keywords", lambda x: x.dropna().iloc[0] if not x.dropna().empty else "")
        ).reset_index()
    else:
        cluster_counts = pd.DataFrame(columns=["cluster", "count", "keywords"])
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Cluster Distribution")
        st.dataframe(cluster_counts.sort_values("count", ascending=False))
    
    with col2:
        with st.expander("🔍 How to Use This Data", expanded=False):
            st.markdown("""
            **For SEO Strategy:**
            - **Large clusters**: Indicate strong topic coverage
            - **Small clusters**: May need more content
            - **Keywords**: Show topic themes for each cluster
            
            **Action Items:**
            - Expand small but important clusters
            - Create content to fill gaps
            - Link related cluster content together
            """)

    with st.expander("🧠 How Clustering Works", expanded=False):
        st.markdown("""
        ### **AI-Powered Content Clustering Process**
        
        **1. Content Understanding (Embeddings)**
        - **Sentence-BERT**: Captures semantic meaning and context
        - **TF-IDF**: Identifies important keywords and phrases
        - **Hybrid Approach**: Combines both for robust similarity measurement
        
        **2. Clustering Algorithm (K-Means with K=7)**
        - Groups content by combined semantic + keyword similarity
        - Automatically identifies optimal topic boundaries
        - Creates balanced clusters representing your content themes
        
        **3. SEO Benefits**
        - **Topic Authority**: Clear thematic organization
        - **Content Gaps**: Identify missing topics in your strategy
        - **Internal Linking**: Connect related content within clusters
        """)
else:
    st.warning("No cluster data available. Run the analysis first.")

st.markdown(f"## 🏆 **Authority & Influence Analysis - {selected_site}**")

st.markdown("""
<div class="info-box">
<strong>💡 SEO Insight:</strong> High-authority pages in your internal link graph should be your primary content for promotion and external link building.
</div>
""", unsafe_allow_html=True)

if not centrality.empty and not nodes.empty:
    # Clean the merge - remove pillar_sim if it doesn't exist
    merge_cols = ["url", "title", "cluster", "cluster_label"]
    if "pillar_sim" in nodes.columns:
        merge_cols.append("pillar_sim")
    merged = centrality.merge(nodes[merge_cols], on="url", how="left")
    top_central = merged.sort_values("pagerank", ascending=False).head(25)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📈 Top Authority Pages (by PageRank)")
        st.dataframe(top_central)
    
    with col2:
        with st.expander("📊 Metrics Explained", expanded=False):
            st.markdown("""
            **🔗 PageRank**
            - Measures page importance in link network
            - Higher = more internal link authority
            - **SEO Use**: Focus promotion on high PageRank pages
            
            **🌉 Betweenness**  
            - Identifies bridge/hub content
            - High = connects different topic clusters
            - **SEO Use**: Great for internal linking strategy
            
            **📍 Closeness**
            - How easily page reaches all others
            - High = central to your content ecosystem
            - **SEO Use**: Ideal for navigation and important CTAs
            
            **🔄 In/Out Degree**
            - Number of internal links pointing to/from page
            - **SEO Use**: Balance link equity distribution
            """)
            
        with st.expander("🚀 How to Apply These Insights", expanded=False):
            st.markdown("""
            **High PageRank Pages:**
            - Promote externally (social, email)
            - Target for external backlinks
            - Use for important product/service promotion
            
            **High Betweenness Pages:**
            - Perfect for topic cluster hub pages
            - Add navigation elements
            - Link to related content in other clusters
            
            **High Closeness Pages:**
            - Ideal for internal search/navigation
            - Place important CTAs here
            - Use for cross-selling content
            """)
else:
    st.warning("No centrality data available. Run the analysis first.")

st.markdown(f"## 🔑 **Keyword Intelligence - {selected_site}**")

st.markdown("""
<div class="info-box">
<strong>💡 SEO Insight:</strong> Understanding your content's keyword profile helps identify content gaps and optimization opportunities.
</div>
""", unsafe_allow_html=True)

if keywords_file.exists() and keywords_file.stat().st_size > 1:
    try:
        kws = pd.read_csv(keywords_file)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🎯 Site-Wide Keyword Analysis")
            st.dataframe(kws.head(50))
        
        with col2:
            with st.expander("📊 TF-IDF Scoring", expanded=False):
                st.markdown("""
                **What is TF-IDF?**
                - **TF**: Term frequency in document
                - **IDF**: Inverse document frequency across site
                - **Combined**: Identifies uniquely important terms
                
                **SEO Applications:**
                - High scores = distinctive content themes
                - Use for meta descriptions and titles
                - Identify content differentiation opportunities
                """)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.info(f"keywords.csv is empty for {selected_site}")
else:
    st.info(f"keywords.csv not found or empty for {selected_site}")

if pillars_file.exists() and pillars_file.stat().st_size > 1:
    try:
        st.markdown("## 🏛️ **Pillar Content Strategy**")
        st.markdown("""
        <div class="success-box">
        <strong>🎯 SEO Strategy:</strong> Pillar pages are your topic authority anchors. These should be your most comprehensive, well-linked content pieces.
        </div>
        """, unsafe_allow_html=True)
        
        pillars = pd.read_csv(pillars_file)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🏆 Auto-Identified Pillar Content")
            st.dataframe(pillars)
        
        with col2:
            with st.expander("🎯 Pillar Strategy", expanded=False):
                st.markdown("""
                **What Makes a Pillar?**
                - Highest semantic relevance to cluster topic
                - Strong internal link authority (PageRank)
                - Comprehensive content coverage
                
                **SEO Action Items:**
                - Update pillars with comprehensive content
                - Link all cluster content to pillars
                - Target pillars for external backlinks
                - Use pillars for topic cluster navigation
                """)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.info(f"pillars.csv is empty for {selected_site}")

if article_kws_file.exists() and article_kws_file.stat().st_size > 1:
    try:
        st.markdown("## 📝 **Article-Level Keyword Analysis**")
        ak = pd.read_csv(article_kws_file)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🔍 Per-Article Keywords (Top Ranking)")
            st.dataframe(ak)
        
        with col2:
            with st.expander("💡 Content Optimization", expanded=False):
                st.markdown("""
                **Use This Data To:**
                - Optimize title tags and headers
                - Identify content expansion opportunities
                - Find internal linking anchor text
                - Discover related topic opportunities
                
                **GEO Benefits:**
                - AI systems understand your content themes
                - Better context for generative AI responses
                - Enhanced semantic search visibility
                """)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.info(f"article_keywords.csv is empty for {selected_site}")
else:
    st.info(f"article_keywords.csv not found or empty for {selected_site}")

st.markdown("## 🔗 **Internal Linking Opportunities**")

st.markdown("""
<div class="warning-box">
<strong>⚡ Action Required:</strong> These AI-generated recommendations can significantly boost your internal link equity distribution and topic authority.
</div>
""", unsafe_allow_html=True)

if recs_file.exists() and recs_file.stat().st_size > 1:
    try:
        recs = pd.read_csv(recs_file)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🎯 Smart Link Recommendations")
            st.dataframe(recs.head(100))
        
        with col2:
            with st.expander("🚀 Implementation Guide", expanded=False):
                st.markdown("""
                **How Recommendations Work:**
                - Based on semantic content similarity
                - Identifies missing high-value connections
                - Prioritizes by relevance score
                
                **Implementation Priority:**
                1. **High similarity** (>0.7): Immediate action
                2. **Medium similarity** (0.5-0.7): Review context
                3. **Lower similarity** (<0.5): Consider topical relevance
                
                **SEO Impact:**
                - Improves topic cluster connectivity
                - Distributes PageRank more effectively
                - Enhances user journey through content
                """)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        st.info(f"recommendations.csv is empty for {selected_site}")
else:
    st.info(f"recommendations.csv not found or empty for {selected_site}")

st.markdown("## 🔍 **Anchor Text Intelligence**")

st.markdown("""
<div class="info-box">
<strong>💡 SEO Insight:</strong> Anchor text analysis reveals how your content connects and what signals you're sending to search engines.
</div>
""", unsafe_allow_html=True)

if not edges.empty:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🔎 Anchor Text Explorer")
        q = st.text_input("Search anchor text contains", "", help="Find specific anchor text patterns across your site")
        if q:
            filt = edges[edges["anchors"].fillna("").str.contains(q, case=False)]
            st.dataframe(filt.head(200))
            st.info(f"Found {len(filt)} links containing '{q}'")
        else:
            st.dataframe(edges.head(200))
    
    with col2:
        with st.expander("📊 Anchor Text Strategy", expanded=False):
            st.markdown("""
            **Optimization Opportunities:**
            - **Generic anchors** ("click here", "read more"): Replace with descriptive text
            - **Exact match**: Good for target keywords
            - **Branded anchors**: Build brand authority
            - **Long-tail anchors**: Support semantic SEO
            
            **GEO Benefits:**
            - Helps AI understand content relationships
            - Provides context for content relevance
            - Improves content discoverability in AI responses
            """)
else:
    st.warning("No edge data available for anchor text analysis.")

st.markdown("## 🕸️ **Interactive Content Network**")

st.markdown("""
<div class="success-box">
<strong>🎯 Network Analysis:</strong> Visualize your content ecosystem. Node size = PageRank authority, colors = topic clusters, connections = internal links.
</div>
""", unsafe_allow_html=True)

if not nodes.empty and not edges.empty:
    # Sidebar controls for better organization
    st.sidebar.subheader("🎛️ Graph Controls")
    
    # Filters
    article_only = st.sidebar.checkbox("📝 Article-only view", value=True, help="Focus on blog content, hide product/navigation pages")
    cluster_options = []
    if "cluster" in nodes.columns and not nodes[nodes["is_article"] == True]["cluster"].dropna().empty:
        cluster_options = sorted([int(c) for c in nodes[nodes["is_article"] == True]["cluster"].dropna().unique()])
    selected_clusters = st.sidebar.multiselect("🎯 Show clusters", options=cluster_options, default=cluster_options, help="Select specific topic clusters to display")
    
    # Layout and edge controls
    layout_choice = st.sidebar.selectbox("🗺️ Layout algorithm", ["UMAP", "Force-directed"], 
                                       index=0 if (base / "umap.csv").exists() else 1,
                                       help="UMAP: AI-organized by content similarity, Force-directed: Traditional network layout")
    hide_cross = st.sidebar.checkbox("🚫 Hide cross-cluster links", value=False, help="Show only links within the same topic cluster")
    max_w = int(edges["weight"].max()) if "weight" in edges.columns and not edges.empty else 5
    min_w = st.sidebar.slider("🔗 Min link strength", 1, max(1, max_w), value=1, help="Filter out weak internal links")
    dynamic_k = 7
    
    # Graph explanation
    with st.expander("📊 How to Read This Network", expanded=False):
        st.markdown("""
        ### **Visual Elements Explained**
        
        **🔵 Node (Page) Properties:**
        - **Size**: PageRank authority (bigger = more important)
        - **Color**: Topic cluster assignment (same color = related content themes)
        - **Position**: Depends on layout choice (see below)
        
        **🔗 Edge (Link) Properties:**
        - **Thickness**: Link frequency/strength between pages
        - **Direction**: Link direction (hover to see anchor text)
        
        **📊 Layout Types & Node Positioning:**
        
        **UMAP Layout** 🎯
        - **Position**: AI-organized by content similarity (SBERT + TF-IDF)
        - **Expected**: Same-colored nodes clustered together spatially
        - **Why scattered colors?**: Dynamic clustering (K=7) may differ from saved clusters
        - **Best for**: Understanding content relationships and topic gaps
        
        **Force-Directed Layout** 🔄  
        - **Position**: Physics-based on link relationships
        - **Expected**: Highly linked pages pulled together
        - **Colors**: Still show topic clusters, but position shows link structure
        - **Best for**: Understanding internal link flow and authority distribution
        
        ### **Color Clustering Explanation**
        If same-colored nodes aren't grouped together in UMAP:
        1. **Dynamic reclustering**: The K=7 clustering may differ from original analysis
        2. **Multi-dimensional complexity**: UMAP reduces 448D to 2D, losing some relationships
        3. **Hybrid signals**: Content similarity (SBERT) + keyword patterns (TF-IDF) create complex groupings
        
        ### **SEO Insights to Look For**
        - **Isolated nodes**: Content needing more internal links
        - **Dense clusters**: Strong topic authority areas  
        - **Bridge nodes**: Content connecting different topics
        - **Central hubs**: High-authority pages for promotion
        - **Color scatter**: Potential content organization opportunities
        """)
    
    # Performance tip
    if len(nodes) > 100:
        st.info("💡 **Performance Tip**: Use filters to focus on specific clusters for better visualization of large sites.")
    
    # Color legend in sidebar
    if cluster_options:
        st.sidebar.subheader("🎨 Cluster Color Legend")
        palette = [
            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
            "#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd","#e6550d","#31a354","#756bb1","#636363",
        ]
        for cluster_id in sorted(cluster_options):
            color = palette[cluster_id % len(palette)]
            st.sidebar.markdown(f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>Cluster {cluster_id}</div>', unsafe_allow_html=True)

    nodes_f = nodes.copy()
    if article_only:
        nodes_f = nodes_f[nodes_f["is_article"] == True]
    if selected_clusters and "cluster" in nodes_f.columns:
        nodes_f = nodes_f[nodes_f["cluster"].isin(selected_clusters)]

    keep_urls = set(nodes_f["url"]) if not nodes_f.empty else set()
    edges_f = edges[edges["source"].isin(keep_urls) & edges["target"].isin(keep_urls)].copy()
    if "weight" in edges_f.columns:
        edges_f = edges_f[edges_f["weight"].fillna(1).astype(float) >= float(min_w)]
    if hide_cross and "cluster" in nodes_f.columns:
        clu_map = nodes_f.set_index("url")["cluster"].to_dict()
        edges_f = edges_f[[clu_map.get(s) == clu_map.get(t) for s, t in zip(edges_f["source"], edges_f["target"])]]

    # Merge centrality for sizing if available
    if not centrality.empty:
        nodes_f = nodes_f.merge(centrality, on="url", how="left")
    else:
        nodes_f["pagerank"] = 0.0

    # Size by pagerank - balanced sizing for readability
    pr = nodes_f["pagerank"].fillna(0.0)
    pr_min, pr_max = float(pr.min()), float(pr.max())
    denom = (pr_max - pr_min) if (pr_max - pr_min) > 1e-12 else 1.0
    sizes = 15.0 + 25.0 * (pr - pr_min) / denom  # Moderate sizing: 15-40 range

    # Color by cluster
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd","#e6550d","#31a354","#756bb1","#636363",
    ]
    def cluster_color(row) -> str:
        if "cluster" in row and not pd.isna(row["cluster"]):
            idx = int(row["cluster"]) % len(palette)
            return palette[idx]
        return "#cccccc"

    # Enhanced network configuration with more space for nodes
    net = Network(
        height="1000px",  # Increased height for more space
        width="100%", 
        directed=True, 
        notebook=False, 
        cdn_resources="in_line",
        bgcolor="#ffffff",
        font_color="black"
    )
    
    # Configure balanced styling for readability and separation
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {
          "size": 12,
          "color": "black",
          "strokeWidth": 1,
          "strokeColor": "white"
        },
        "scaling": {
          "min": 15,
          "max": 40
        },
        "margin": 10
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
        "color": {"opacity": 0.6},
        "smooth": {"enabled": true, "type": "continuous"},
        "width": 1.5
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 300},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.05,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.2,
          "avoidOverlap": 0.5
        }
      }
    }
    """)
    
    use_umap_layout = layout_choice == "UMAP" and (base / "umap.csv").exists()
    umap_pos = {}
    cluster_keywords = {}
    
    # Status indicator and layout configuration
    if use_umap_layout:
        st.success("🎯 Using AI-optimized UMAP layout based on content similarity")
        st.info("📍 **Color Legend**: Same colors = same topic cluster. Position = content similarity.")
    else:
        st.info("🔄 Using force-directed layout based on link relationships")
        st.info("📍 **Color Legend**: Same colors = same topic cluster. Position = link relationships.")
        # Physics is already configured in set_options above
    if use_umap_layout:
        umap_df = pd.read_csv(base / "umap.csv")
        umap_df["cluster"] = umap_df["cluster"].astype(int)
        # If dynamic K != saved labels, recompute labels on the fly using saved hybrid embeddings
        emb_path = base / "embeddings_hybrid.npy"
        if emb_path.exists():
            try:
                import numpy as np
                from sklearn.cluster import KMeans
                embs = np.load(emb_path)
                km = KMeans(n_clusters=int(dynamic_k), n_init=10, random_state=42)
                dyn_labels = km.fit_predict(embs)
                umap_df["cluster"] = dyn_labels.astype(int)
            except Exception:
                pass
        # Normalize coords to centered pixel space
        minx, maxx = float(umap_df["x"].min()), float(umap_df["x"].max())
        miny, maxy = float(umap_df["y"].min()), float(umap_df["y"].max())
        rangex, rangey = maxx - minx + 1e-6, maxy - miny + 1e-6
        scale = 900.0 / max(rangex, rangey)
        def to_px(x, y):
            return ((x - (minx + rangex / 2.0)) * scale, -1.0 * (y - (miny + rangey / 2.0)) * scale)
        for _, r in umap_df.iterrows():
            umap_pos[r["url"]] = to_px(float(r["x"]), float(r["y"]))
        ck_path = base / "cluster_keywords.csv"
        if ck_path.exists():
            ck = pd.read_csv(ck_path)
            if "cluster" in ck.columns and "keywords" in ck.columns:
                cluster_keywords = {int(c): str(k) for c, k in zip(ck["cluster"], ck["keywords"])}
        # Override node clusters with dynamic clusters for coloring/grouping
        if not umap_df.empty and "cluster" in umap_df.columns:
            dyn_map = umap_df.set_index("url")["cluster"].to_dict()
            # Fix for future warning: handle NaN values explicitly
            mapped_clusters = nodes_f["url"].map(dyn_map)
            nodes_f["cluster"] = mapped_clusters.where(mapped_clusters.notna(), nodes_f["cluster"])

    for (_, row), size in zip(nodes_f.iterrows(), sizes):
        title = row.get("title", "") or ""
        cluster = row.get("cluster")
        clabel = row.get("cluster_label")
        
        # Enhanced tooltip with SEO metrics
        tooltip_parts = []
        if title:
            tooltip_parts.append(f"<b>📝 {title}</b>")
        if cluster is not None and not pd.isna(cluster):
            tooltip_parts.append(f"<b>🎯 Cluster:</b> {int(cluster)}")
        if clabel:
            tooltip_parts.append(f"<b>🏷️ Topic:</b> {clabel}")
        
        # Add centrality metrics to tooltip if available
        if not centrality.empty:
            url = row['url']
            cent_row = centrality[centrality['url'] == url]
            if not cent_row.empty:
                pr = cent_row.iloc[0].get('pagerank', 0)
                indeg = cent_row.iloc[0].get('in_degree', 0) 
                outdeg = cent_row.iloc[0].get('out_degree', 0)
                tooltip_parts.append(f"<b>📊 PageRank:</b> {pr:.3f}")
                tooltip_parts.append(f"<b>🔗 Links:</b> {indeg} in, {outdeg} out")
        
        tooltip_parts.append(f"<b>🌐 URL:</b> {row['url']}")
        tooltip = "<br/>".join(tooltip_parts)

        add_kwargs = {}
        if use_umap_layout and row["url"] in umap_pos:
            px, py = umap_pos[row["url"]]
            add_kwargs.update({"x": float(px), "y": float(py), "physics": False, "fixed": True})
        # Make labels appropriately sized for readability
        # Handle NaN values and ensure title is a string
        title_str = str(title) if pd.notna(title) and title else ""
        display_label = title_str[:35] + "…" if title_str and len(title_str) > 35 else (title_str or row["url"])
        
        net.add_node(
            row["url"],
            label=display_label,
            title=tooltip,
            color=cluster_color(row),
            value=float(size),
            **add_kwargs,
        )

    for _, e in edges_f.iterrows():
        net.add_edge(e["source"], e["target"], value=float(e.get("weight", 1) or 1), title=e.get("anchors", ""))

    # Add cluster centroid nodes when using UMAP to visually group pages
    if use_umap_layout and not nodes_f.empty and "cluster" in nodes_f.columns:
        for c, grp in nodes_f.groupby("cluster"):
            pts = [umap_pos[u] for u in grp["url"] if u in umap_pos]
            if not pts:
                continue
            cx = float(sum(p[0] for p in pts) / len(pts))
            cy = float(sum(p[1] for p in pts) / len(pts))
            label_terms = cluster_keywords.get(int(c), grp["cluster_label"].dropna().iloc[0] if "cluster_label" in grp.columns and not grp["cluster_label"].dropna().empty else "")
            net.add_node(
                f"cluster:{int(c)}",
                label=f"C{int(c)}",  # Shorter label
                title=f"Cluster {int(c)}\n{label_terms}",
                color="#000000",
                font={"color": "#ffffff", "size": 14, "strokeWidth": 1, "strokeColor": "#000000"},
                value=50,  # Moderate cluster centroid size
                x=cx,
                y=cy,
                physics=False,
                fixed=True,
                shape="ellipse",
            )

    html = net.generate_html(notebook=False)
    
    # Display network with loading message
    with st.spinner("🎨 Rendering interactive network visualization..."):
        components.html(html, height=1020, scrolling=True)
    
    # Post-graph insights
    st.markdown("""
    <div class="info-box">
    <strong>💡 Quick Analysis Tips:</strong>
    <ul>
    <li><strong>Large nodes</strong> = High authority (focus external promotion here)</li>
    <li><strong>Connected clusters</strong> = Good topic flow (maintain/strengthen)</li>
    <li><strong>Isolated nodes</strong> = Link opportunities (connect to relevant content)</li>
    <li><strong>Dense clusters</strong> = Strong topic authority (leverage for expertise)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.warning("⚠️ No graph data available. Please run the analysis pipeline first.")
    st.markdown("""
    **To generate network data:**
    1. Run the CLI tool: `python -m seo_graph.cli all --seed YOUR_URL`
    2. Wait for analysis to complete
    3. Refresh this dashboard
    """)

st.markdown("## 📚 **Technical Documentation & Methodology**")

with st.expander("🔬 Complete Technical Glossary", expanded=False):
    st.markdown("""
    ### **🧠 AI & Machine Learning Components**
    
    **Embeddings (SBERT)**
    - Model: Sentence-BERT `all-MiniLM-L6-v2` (384 dimensions)
    - Purpose: Captures semantic meaning and context
    - Processing: L2-normalized for cosine similarity calculations
    - **SEO Value**: Understands content meaning beyond keywords
    
    **TF-IDF (Term Frequency-Inverse Document Frequency)**
    - Configuration: 1-2 gram terms, English stopwords, min_df=2
    - Purpose: Identifies statistically important keywords
    - **SEO Value**: Reveals distinctive content themes and keyword opportunities
    
    **Hybrid Embeddings**
    - Method: Concatenation of SBERT (384d) + TF-IDF SVD (64d)
    - Blending: α=0.5 for balanced semantic + keyword signals
    - **SEO Value**: Robust content similarity that considers both meaning and keywords
    
    ### **📊 Clustering & Analysis**
    
    **K-Means Clustering**
    - Configuration: K=7 clusters over hybrid embeddings
    - Purpose: Groups content by combined semantic + keyword similarity
    - **SEO Value**: Identifies natural topic boundaries for content strategy
    
    **UMAP (Uniform Manifold Approximation)**
    - Parameters: n_neighbors=15, min_dist=0.1, 2D projection
    - Purpose: Spatial visualization of content relationships
    - **SEO Value**: Visual content gaps and cluster organization
    
    ### **🔗 Network Analysis Metrics**
    
    **PageRank**
    - Algorithm: Google's PageRank with damping factor α=0.85
    - Weighted by: Internal link frequency and anchor text repetition
    - **SEO Insight**: Internal authority and link equity distribution
    - **Action**: Promote high PageRank pages externally
    
    **Betweenness Centrality**
    - Calculation: Proportion of shortest paths passing through each node
    - **SEO Insight**: Identifies bridge content connecting different topics
    - **Action**: Optimize these pages for internal navigation and topic linking
    
    **Closeness Centrality**  
    - Calculation: Inverse of average shortest-path distance to all other nodes
    - **SEO Insight**: How easily users/crawlers can reach all content from this page
    - **Action**: Ideal pages for important CTAs and navigation elements
    
    **Degree Centrality**
    - In-degree: Number of internal links pointing to the page
    - Out-degree: Number of internal links from the page to others
    - **SEO Insight**: Link equity flow and content connectivity
    - **Action**: Balance link distribution to avoid over/under-linking
    
    ### **🎯 Content Strategy Metrics**
    
    **Pillar Content Identification**
    - Method: Highest semantic relevance to cluster centroid + PageRank authority
    - Filtering: Excludes author pages, requires minimum content length
    - **SEO Value**: Identifies natural topic authority pages
    - **Action**: Develop these as comprehensive topic hubs
    
    **Similarity Scoring**
    - Semantic: SBERT cosine similarity for content meaning
    - Keyword: TF-IDF cosine similarity for term overlap  
    - Blended: 0.6 semantic + 0.4 keyword for balanced relevance
    - **SEO Value**: Finds related content for internal linking
    """)

with st.expander("🚀 SEO & GEO Implementation Playbook", expanded=False):
    st.markdown("""
    ### **Immediate Actions**
    1. **High-Priority Internal Links**: Implement similarity >0.7 recommendations
    2. **Authority Page Promotion**: Focus external marketing on top PageRank pages  
    3. **Isolated Content**: Add internal links to disconnected pages
    
    ### **Content Strategy**
    1. **Cluster Expansion**: Create content for small but important clusters
    2. **Pillar Enhancement**: Expand pillar pages with comprehensive coverage
    3. **Anchor Text Optimization**: Replace generic anchors with descriptive text
    
    ### **Long-term Optimization**
    1. **Topic Gap Analysis**: Identify missing content themes
    2. **Authority Building**: Build external links to high-centrality pages
    3. **Content Refresh**: Update low-authority content in important clusters
    
    ### **Generative Engine Optimization (GEO)**
    1. **Semantic Coherence**: Ensure cluster content forms logical knowledge graphs
    2. **Context Richness**: Interconnect related concepts across clusters  
    3. **Authority Signals**: Build clear content hierarchies AI can understand
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔄 Data Refresh**")
    st.caption("Re-run CLI analysis to update data")
with col2:
    st.markdown("**📊 Export Data**") 
    st.caption("All analysis exported to CSV files")
with col3:
    st.markdown("**🎯 SEO Impact**")
    st.caption("Apply insights for measurable SEO improvements")

st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <strong>SEO Graph Explorer v2.0</strong> | Advanced Content Analysis & Link Intelligence
</div>
""", unsafe_allow_html=True)
