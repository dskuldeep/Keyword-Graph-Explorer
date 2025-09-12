# Multi-Path Configuration Quick Reference

## 🚀 Quick Start

### 1. Edit Configuration
```bash
nano blog_config.json
```

### 2. Run Analysis
```bash
python -m seo_graph.cli all-blogs
```

### 3. View Results
```bash
streamlit run src/seo_graph/app.py --server.port 8502
```

## 📋 Minimal Configuration Template

```json
{
  "blog_paths": [
    {
      "name": "your-site.com/blog",
      "url": "https://your-site.com/blog",
      "domain": "your-site.com",
      "output_dir": "your_blog"
    }
  ],
  "article_detection": {
    "url_patterns": ["/blog/", "/articles/", "/posts/"],
    "exclude_patterns": ["/author/", "/tag/", "/category/", "/page/"]
  },
  "crawl_settings": {
    "max_pages": 100,
    "max_depth": 3
  }
}
```

## 🎯 Common URL Patterns

### Blog Platforms
- **WordPress**: `/blog/`, `/news/`, `/category/`
- **Medium**: `/@username/`, `/story/`
- **Ghost**: `/blog/`, `/tag/`
- **Custom**: `/articles/`, `/posts/`, `/insights/`

### News Sites
- **News**: `/news/`, `/stories/`, `/reports/`
- **Tech**: `/tech/`, `/technology/`
- **Business**: `/business/`, `/finance/`

### Corporate Sites
- **Resources**: `/resources/`, `/guides/`, `/whitepapers/`
- **Case Studies**: `/case-studies/`, `/success-stories/`
- **Updates**: `/updates/`, `/announcements/`

## ⚙️ Configuration Fields

### Required Fields
- `name`: Display name in dropdown
- `url`: Full blog URL
- `domain`: Domain for crawling
- `output_dir`: Output directory name

### Optional Fields
- `description`: Human-readable description

### Article Detection (GLOBAL - applies to ALL blog paths)
- `url_patterns`: URL patterns for articles (must match at least one)
- `exclude_patterns`: **GLOBAL EXCLUSION** - patterns to exclude from ALL paths
- `title_exclude_patterns`: Title patterns to exclude (case-insensitive)
- `min_title_length`: Minimum title length (default: 10)

### Crawl Settings
- `max_pages`: Max pages to crawl (default: 100)
- `max_depth`: Max crawl depth (default: 3)
- `timeout`: Request timeout in seconds (default: 30)
- `delay`: Delay between requests (default: 1)

## 🚫 Exclude Patterns (GLOBAL RULES)

**These patterns apply to ALL blog paths and will exclude URLs from article analysis:**

```json
"exclude_patterns": [
  "/author/", "/authors/",     // Author pages
  "/tag/", "/tags/",           // Tag pages  
  "/category/", "/categories/", // Category pages
  "/page/",                    // Pagination
  "/search", "/feed", "/rss",  // Search & feeds
  "/about/", "/contact/",      // Static pages
  "/privacy/", "/terms/"       // Legal pages
]
```

**Priority**: Exclude patterns override URL patterns. If a URL matches both, it's EXCLUDED.

## 🔧 Troubleshooting

### Too Many Non-Articles
Add to `exclude_patterns`:
```json
"/author/", "/tag/", "/category/", "/page/", "/search", "/feed", "/rss"
```

### Missing Articles
Add to `url_patterns`:
```json
"/blog/", "/articles/", "/posts/", "/news/", "/insights/", "/updates/"
```

### Performance Issues
Reduce settings:
```json
{
  "max_pages": 50,
  "max_depth": 2,
  "timeout": 15,
  "delay": 2
}
```

## 📊 CLI Commands

```bash
# Process all configured paths
python -m seo_graph.cli all-blogs

# Process with custom config
python -m seo_graph.cli all-blogs --config my_config.json

# Process single path
python -m seo_graph.cli all \
  --seed https://example.com/blog \
  --domain example.com \
  --out output/example_blog
```

## 📁 Output Structure

Each path creates:
```
output/your_blog/
├── nodes.csv              # All pages
├── nodes_articles.csv     # Articles only
├── edges.csv              # All links
├── centrality.csv         # PageRank metrics
├── keywords.csv           # Keyword analysis
├── clusters.csv           # Topic clusters
├── recommendations.csv    # Link suggestions
└── graph.html            # Interactive visualization
```

## 🎨 UI Features

- **Dropdown Selector**: Switch between paths
- **Path Context**: Header shows current path
- **Independent Data**: Each path has separate analysis
- **Error Handling**: Clear messages for missing data
