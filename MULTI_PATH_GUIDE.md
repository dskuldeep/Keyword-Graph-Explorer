# Multi-Path Analysis Configuration Guide

## Overview

The SEO Graph Explorer supports analyzing multiple blog paths/sites independently through a flexible configuration system. Each path maintains its own analysis data and can be switched between using the dropdown selector in the sidebar.

## Configuration File Structure

The system uses `blog_config.json` to define all blog paths and their analysis settings. Here's the complete structure:

```json
{
  "blog_paths": [
    {
      "name": "Display Name",
      "url": "https://example.com/blog",
      "domain": "example.com",
      "output_dir": "output_directory_name",
      "description": "Optional description"
    }
  ],
  "article_detection": {
    "url_patterns": ["/blog/", "/articles/", "/posts/"],
    "exclude_patterns": ["/author/", "/tag/", "/category/"],
    "title_exclude_patterns": ["blog", "author", "tag"],
    "min_title_length": 10,
    "max_depth": 3
  },
  "crawl_settings": {
    "max_pages": 100,
    "max_depth": 3,
    "timeout": 30,
    "delay": 1
  }
}
```

## Detailed Configuration Options

### 1. Blog Paths Configuration

Each blog path requires the following fields:

#### Required Fields:
- **`name`**: Display name shown in the dropdown (e.g., "example.com/blog")
- **`url`**: Full URL to the blog section (e.g., "https://example.com/blog")
- **`domain`**: Domain for crawling (e.g., "example.com")
- **`output_dir`**: Directory name for output files (e.g., "example_blog")

#### Optional Fields:
- **`description`**: Human-readable description of the blog section

#### Example Blog Paths:
```json
{
  "blog_paths": [
    {
      "name": "getmaxim.ai/articles",
      "url": "https://getmaxim.ai/articles",
      "domain": "getmaxim.ai",
      "output_dir": "getmaxim",
      "description": "Main articles section"
    },
    {
      "name": "getmaxim.ai/blog",
      "url": "https://getmaxim.ai/blog", 
      "domain": "getmaxim.ai",
      "output_dir": "getmaxim_blog",
      "description": "Blog posts and updates"
    },
    {
      "name": "company.com/insights",
      "url": "https://company.com/insights",
      "domain": "company.com",
      "output_dir": "company_insights",
      "description": "Industry insights and thought leadership"
    }
  ]
}
```

### 2. Article Detection Configuration

Controls how the system identifies article pages vs. non-article pages. **These settings apply globally to ALL blog paths** defined in the configuration.

#### `url_patterns` (Array of Strings)
URL patterns that indicate article content. A URL must contain at least one of these patterns to be considered an article:
```json
"url_patterns": [
  "/blog/",
  "/articles/", 
  "/posts/",
  "/news/",
  "/insights/",
  "/updates/",
  "/case-studies/",
  "/whitepapers/"
]
```

#### `exclude_patterns` (Array of Strings)
**GLOBAL EXCLUSION RULES** - These patterns are excluded from article analysis across ALL blog paths. If a URL contains any of these patterns, it will NOT be treated as an article, regardless of which blog path it belongs to:

```json
"exclude_patterns": [
  "/author/",           // Author profile pages
  "/authors/",          // Author listing pages
  "/tag/",              // Tag pages
  "/tags/",             // Tag listing pages
  "/category/",         // Category pages
  "/categories/",       // Category listing pages
  "/page/",             // Pagination pages (e.g., /blog/page/2/)
  "/archive",           // Archive pages
  "/search",            // Search pages
  "/feed",              // RSS/Atom feeds
  "/rss",               // RSS feeds
  "/sitemap",           // Sitemap pages
  "/about/",            // About pages
  "/contact/",          // Contact pages
  "/privacy/",          // Privacy policy
  "/terms/",            // Terms of service
  "/subscribe/",        // Newsletter signup
  "/newsletter/",       // Newsletter pages
  "/advertise/",        // Advertising pages
  "/jobs/",             // Job listings
  "/careers/"           // Career pages
]
```

**How Exclude Patterns Work:**
1. **Global Application**: These patterns apply to ALL blog paths in your configuration
2. **Priority Override**: If a URL matches both `url_patterns` AND `exclude_patterns`, it will be EXCLUDED
3. **Case Insensitive**: Matching is case-insensitive
4. **Partial Matching**: Patterns match anywhere in the URL path

**Examples:**
- ✅ `https://example.com/blog/my-article/` → **INCLUDED** (matches `/blog/`, no exclusions)
- ❌ `https://example.com/blog/author/john-doe/` → **EXCLUDED** (matches `/author/`)
- ❌ `https://example.com/articles/tag/seo/` → **EXCLUDED** (matches `/tag/`)
- ❌ `https://example.com/blog/page/2/` → **EXCLUDED** (matches `/page/`)

## Article Detection Logic Flow

The system follows this step-by-step process to determine if a URL is an article:

### Step 1: Check URL Patterns
```python
# URL must contain at least one pattern from url_patterns
url = "https://example.com/blog/my-article/"
url_patterns = ["/blog/", "/articles/", "/posts/"]

# Check: Does URL contain any pattern?
matches_pattern = any(pattern in url.lower() for pattern in url_patterns)
# Result: True (contains "/blog/")
```

### Step 2: Check Exclude Patterns
```python
# If URL matches exclude_patterns, it's automatically excluded
exclude_patterns = ["/author/", "/tag/", "/category/", "/page/"]

# Check: Does URL contain any exclusion pattern?
is_excluded = any(pattern in url.lower() for pattern in exclude_patterns)
# Result: False (no exclusion patterns found)
```

### Step 3: Final Decision
```python
# Article if: matches URL pattern AND not excluded
is_article = matches_pattern and not is_excluded
# Result: True (URL is an article)
```

### Real Examples:

| URL | URL Pattern Match | Exclude Pattern Match | Final Result | Reason |
|-----|------------------|----------------------|--------------|---------|
| `https://site.com/blog/article/` | ✅ `/blog/` | ❌ None | **ARTICLE** | Matches pattern, no exclusions |
| `https://site.com/articles/news/` | ✅ `/articles/` | ❌ None | **ARTICLE** | Matches pattern, no exclusions |
| `https://site.com/blog/author/john/` | ✅ `/blog/` | ✅ `/author/` | **NOT ARTICLE** | Excluded despite matching pattern |
| `https://site.com/articles/tag/seo/` | ✅ `/articles/` | ✅ `/tag/` | **NOT ARTICLE** | Excluded despite matching pattern |
| `https://site.com/blog/page/2/` | ✅ `/blog/` | ✅ `/page/` | **NOT ARTICLE** | Excluded despite matching pattern |
| `https://site.com/about/` | ❌ No match | ❌ None | **NOT ARTICLE** | Doesn't match any URL pattern |
| `https://site.com/contact/` | ❌ No match | ❌ None | **NOT ARTICLE** | Doesn't match any URL pattern |

#### `title_exclude_patterns` (Array of Strings)
Title patterns to exclude (case-insensitive):
```json
"title_exclude_patterns": [
  "blog",
  "author", 
  "tag",
  "category",
  "search",
  "page",
  "archive"
]
```

#### `min_title_length` (Integer)
Minimum title length for articles (default: 10):
```json
"min_title_length": 10
```

#### `max_depth` (Integer)
Maximum crawl depth for article detection (default: 3):
```json
"max_depth": 3
```

### 3. Crawl Settings Configuration

Controls the crawling behavior:

#### `max_pages` (Integer)
Maximum number of pages to crawl per blog path:
```json
"max_pages": 100
```

#### `max_depth` (Integer)
Maximum crawl depth from the seed URL:
```json
"max_depth": 3
```

#### `timeout` (Integer)
Request timeout in seconds:
```json
"timeout": 30
```

#### `delay` (Integer)
Delay between requests in seconds:
```json
"delay": 1
```

## Complete Configuration Examples

### Example 1: Multi-Blog Company
```json
{
  "blog_paths": [
    {
      "name": "company.com/blog",
      "url": "https://company.com/blog",
      "domain": "company.com",
      "output_dir": "company_blog",
      "description": "Main company blog"
    },
    {
      "name": "company.com/resources",
      "url": "https://company.com/resources",
      "domain": "company.com", 
      "output_dir": "company_resources",
      "description": "Resource center and guides"
    },
    {
      "name": "company.com/case-studies",
      "url": "https://company.com/case-studies",
      "domain": "company.com",
      "output_dir": "company_cases",
      "description": "Customer case studies"
    }
  ],
  "article_detection": {
    "url_patterns": [
      "/blog/",
      "/resources/",
      "/case-studies/",
      "/guides/",
      "/whitepapers/"
    ],
    "exclude_patterns": [
      "/author/",
      "/tag/",
      "/category/",
      "/page/",
      "/search",
      "/contact/",
      "/about/"
    ],
    "title_exclude_patterns": [
      "blog",
      "author",
      "tag",
      "category",
      "search"
    ],
    "min_title_length": 15,
    "max_depth": 2
  },
  "crawl_settings": {
    "max_pages": 200,
    "max_depth": 2,
    "timeout": 30,
    "delay": 1
  }
}
```

### Example 2: News Website
```json
{
  "blog_paths": [
    {
      "name": "news.com/tech",
      "url": "https://news.com/tech",
      "domain": "news.com",
      "output_dir": "news_tech",
      "description": "Technology news section"
    },
    {
      "name": "news.com/business",
      "url": "https://news.com/business", 
      "domain": "news.com",
      "output_dir": "news_business",
      "description": "Business news section"
    }
  ],
  "article_detection": {
    "url_patterns": [
      "/tech/",
      "/business/",
      "/news/",
      "/articles/"
    ],
    "exclude_patterns": [
      "/author/",
      "/tag/",
      "/category/",
      "/page/",
      "/search",
      "/subscribe/",
      "/advertise/"
    ],
    "title_exclude_patterns": [
      "news",
      "author",
      "tag",
      "category"
    ],
    "min_title_length": 20,
    "max_depth": 3
  },
  "crawl_settings": {
    "max_pages": 500,
    "max_depth": 3,
    "timeout": 20,
    "delay": 0.5
  }
}
```

## How to Use the Configuration

### 1. Create/Edit Configuration File
```bash
# Edit the configuration file
nano blog_config.json

# Or create from scratch
cp blog_config.json blog_config.json.backup
```

### 2. Run Analysis for All Paths
```bash
# Process all configured blog paths
python -m seo_graph.cli all-blogs

# Or specify a custom config file
python -m seo_graph.cli all-blogs --config my_custom_config.json
```

### 3. Run Analysis for Single Path
```bash
# Process a single path (useful for testing)
python -m seo_graph.cli all \
  --seed https://example.com/blog \
  --domain example.com \
  --out output/example_blog \
  --max-pages 100
```

### 4. View Results in App
```bash
# Start the Streamlit app
streamlit run src/seo_graph/app.py --server.port 8502
```

## Data Structure

Each configured path creates its own directory structure:
```
output/
├── getmaxim/                    # getmaxim.ai/articles
│   ├── nodes.csv               # All discovered pages
│   ├── nodes_articles.csv      # Article pages only
│   ├── edges.csv               # All internal links
│   ├── edges_articles.csv      # Links between articles
│   ├── centrality.csv          # PageRank and centrality metrics
│   ├── keywords.csv            # Site-wide keyword analysis
│   ├── article_keywords.csv    # Per-article keywords
│   ├── cluster_keywords.csv    # Keywords per topic cluster
│   ├── pillars.csv             # Identified pillar content
│   ├── recommendations.csv     # Internal linking suggestions
│   ├── umap.csv                # 2D visualization coordinates
│   ├── embeddings_sbert.npy    # Semantic embeddings
│   ├── embeddings_hybrid.npy   # Combined embeddings
│   ├── graph.html              # Interactive network visualization
│   └── crawl.json              # Raw crawl data
├── getmaxim_blog/              # getmaxim.ai/blog
│   └── [same structure as above]
└── company_insights/           # company.com/insights
    └── [same structure as above]
```

## Current Configuration Breakdown

Based on your current `blog_config.json`, here's what each exclude pattern does:

### Current Exclude Patterns:
```json
"exclude_patterns": [
  "/author/",     // Excludes: /blog/author/john-doe/, /articles/author/sarah/
  "/tag/",        // Excludes: /blog/tag/seo/, /articles/tag/marketing/
  "/category/",   // Excludes: /blog/category/tech/, /articles/category/business/
  "/page/",       // Excludes: /blog/page/2/, /articles/page/3/
  "/search",      // Excludes: /blog/search, /articles/search?q=keyword
  "/feed",        // Excludes: /blog/feed/, /articles/feed.xml
  "/rss",         // Excludes: /blog/rss/, /articles/rss.xml
  "/sitemap"      // Excludes: /blog/sitemap.xml, /articles/sitemap/
]
```

### What This Means for Your Blog Paths:

#### For `getmaxim.ai/articles`:
- ✅ **INCLUDED**: `https://getmaxim.ai/articles/any-article-title/`
- ❌ **EXCLUDED**: `https://getmaxim.ai/articles/author/john/`
- ❌ **EXCLUDED**: `https://getmaxim.ai/articles/tag/seo/`
- ❌ **EXCLUDED**: `https://getmaxim.ai/articles/page/2/`

#### For `getmaxim.ai/blog`:
- ✅ **INCLUDED**: `https://getmaxim.ai/blog/any-blog-post/`
- ❌ **EXCLUDED**: `https://getmaxim.ai/blog/author/sarah/`
- ❌ **EXCLUDED**: `https://getmaxim.ai/blog/tag/marketing/`
- ❌ **EXCLUDED**: `https://getmaxim.ai/blog/page/3/`

## Advanced Configuration Tips

### 1. Optimizing Article Detection
- **Start broad**: Include common blog patterns in `url_patterns`
- **Refine exclusions**: Add specific patterns to `exclude_patterns` as needed
- **Test incrementally**: Run analysis on small sections first

### 2. Performance Tuning
- **Adjust `max_pages`**: Start with 50-100, increase as needed
- **Control `max_depth`**: Deeper crawls take longer but find more content
- **Set appropriate `delay`**: Respectful crawling (1-2 seconds recommended)

### 3. Handling Different Blog Structures
- **WordPress**: Common patterns include `/blog/`, `/news/`, `/category/`
- **Custom CMS**: May use `/articles/`, `/posts/`, `/content/`
- **News Sites**: Often use `/news/`, `/stories/`, `/reports/`

### 4. Troubleshooting Common Issues

#### Too Many Non-Articles Detected
```json
"exclude_patterns": [
  "/author/", "/tag/", "/category/", "/page/",
  "/search", "/feed", "/rss", "/sitemap",
  "/about/", "/contact/", "/privacy/", "/terms/",
  "/subscribe/", "/newsletter/", "/advertise/"
]
```

#### Missing Articles
```json
"url_patterns": [
  "/blog/", "/articles/", "/posts/", "/news/",
  "/insights/", "/updates/", "/case-studies/",
  "/whitepapers/", "/guides/", "/tutorials/"
]
```

#### Performance Issues
```json
"crawl_settings": {
  "max_pages": 50,        // Reduce for faster processing
  "max_depth": 2,         // Limit crawl depth
  "timeout": 15,          // Reduce timeout
  "delay": 2              // Increase delay for slower sites
}
```

## UI Features

- **Dynamic Path Loading**: App automatically loads all configured paths
- **Path Context**: Header shows current analysis path
- **Data Status**: Clear indicators when data is missing
- **Independent Analysis**: Each path maintains separate metrics and visualizations
- **Easy Switching**: Dropdown selector for quick path switching

## Benefits

- **Scalable**: Add unlimited blog paths
- **Flexible**: Works with any blog structure
- **Independent**: Each path analyzed separately
- **Comparable**: Easy to compare different sections
- **Configurable**: Fine-tune detection and crawling behavior
