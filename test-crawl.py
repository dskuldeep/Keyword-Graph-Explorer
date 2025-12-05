# crawl_only.py
from src.seo_graph.crawler import crawl_site
import json
from pathlib import Path
from dataclasses import asdict

pages = crawl_site(
    seed_url='https://www.braintrust.dev/blog',
    allowed_domain='braintrust.dev',
    max_pages=1000,
    max_depth=4,
    sitemap_url='https://www.braintrust.dev/sitemap.xml',
    focus_prefix='https://www.braintrust.dev/blog',
    enable_js_discovery=True,
    enable_pagination_discovery=True,
)

out_dir = Path('test/braintrust_blog')
out_dir.mkdir(parents=True, exist_ok=True)
crawl_path = out_dir / 'crawl.json'
with crawl_path.open('w') as f:
    json.dump({k: asdict(v) for k, v in pages.items()}, f, indent=2)

print(f'✅ Crawled {len(pages)} pages')
print(f'📁 Saved to {crawl_path}')