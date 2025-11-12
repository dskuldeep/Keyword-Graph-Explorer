from __future__ import annotations

import re
import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse
import tldextract
import trafilatura
import xml.etree.ElementTree as ET
import gzip
from io import BytesIO


@dataclass
class Link:
    href: str
    anchor: str


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: List[Link]
    depth: int


def is_internal_url(url: str, allowed_domain: str) -> bool:
    try:
        netloc = urlparse(url).netloc
        if not netloc:
            return True
        extracted = tldextract.extract(netloc)
        domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        return domain == allowed_domain
    except Exception:
        return False


def clean_anchor(text: Optional[str]) -> str:
    if not text:
        return ""
    txt = re.sub(r"\s+", " ", text).strip()
    return txt[:500]


def fetch_page(session: requests.Session, url: str, timeout: float = 15.0) -> Tuple[str, str, List[Link]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    resp = session.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    links: List[Link] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        anchor = clean_anchor(a.get_text(" ", strip=True))
        if href:
            links.append(Link(href=href, anchor=anchor))

    # Prefer trafilatura text extraction
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    text = downloaded or ""

    return title, text, links


def discover_pagination_urls(session: requests.Session, base_url: str, allowed_domain: str) -> List[str]:
    """Discover pagination URLs by checking common patterns and API endpoints."""
    discovered_urls = []
    
    try:
        # Check for common pagination patterns
        pagination_patterns = [
            f"{base_url}?page={{}}",
            f"{base_url}/page/{{}}",
            f"{base_url}/{{}}",
            f"{base_url}?p={{}}",
            f"{base_url}?offset={{}}",
        ]
        
        # Check first few pages
        for pattern in pagination_patterns:
            for page_num in range(1, 6):  # Check first 5 pages
                test_url = pattern.format(page_num)
                try:
                    resp = session.head(test_url, timeout=5, allow_redirects=True)
                    if resp.status_code == 200 and is_internal_url(test_url, allowed_domain):
                        discovered_urls.append(test_url)
                except:
                    continue
        
        # Check for API endpoints that might return blog data
        api_patterns = [
            f"{base_url}/api/posts",
            f"{base_url}/api/articles", 
            f"{base_url}/api/blog",
            f"{base_url}/wp-json/wp/v2/posts",
            f"{base_url}/ghost/api/v3/content/posts",
        ]
        
        for api_url in api_patterns:
            try:
                resp = session.get(api_url, timeout=5)
                if resp.status_code == 200:
                    # Try to extract URLs from JSON response
                    try:
                        data = resp.json()
                        if isinstance(data, dict) and 'posts' in data:
                            for post in data['posts']:
                                if 'url' in post:
                                    discovered_urls.append(post['url'])
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'url' in item:
                                    discovered_urls.append(item['url'])
                    except:
                        pass
            except:
                continue
                
    except Exception:
        pass
    
    return list(set(discovered_urls))


def extract_js_links(html: str, base_url: str) -> List[str]:
    """Extract links from JavaScript code in HTML."""
    links = []
    
    # Look for common patterns in JavaScript
    patterns = [
        r'["\']([^"\']*\/blog\/[^"\']*)["\']',  # Blog URLs in quotes
        r'href\s*[:=]\s*["\']([^"\']+)["\']',   # href assignments
        r'url\s*[:=]\s*["\']([^"\']+)["\']',    # url assignments
        r'["\']([^"\']*\/page\/[^"\']*)["\']',  # Pagination URLs
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        for match in matches:
            if match.startswith('/'):
                full_url = urljoin(base_url, match)
                links.append(full_url)
            elif match.startswith('http'):
                links.append(match)
    
    return links


def discover_dynamic_content(session: requests.Session, base_url: str, allowed_domain: str) -> List[str]:
    """Discover content that might be loaded dynamically via JavaScript or AJAX."""
    discovered_urls = []
    
    try:
        # Get the main page
        resp = session.get(base_url, timeout=10)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract links from JavaScript
        js_links = extract_js_links(html, base_url)
        for link in js_links:
            if is_internal_url(link, allowed_domain):
                discovered_urls.append(link)
        
        # Look for data attributes that might contain URLs
        for element in soup.find_all(attrs={"data-url": True}):
            url = element.get("data-url")
            if url and is_internal_url(url, allowed_domain):
                discovered_urls.append(url)
        
        # Look for JSON-LD structured data
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and "url" in data:
                    discovered_urls.append(data["url"])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "url" in item:
                            discovered_urls.append(item["url"])
            except:
                continue
        
        # Check for meta tags with URLs
        for meta in soup.find_all("meta", property="og:url"):
            content = meta.get("content")
            if content and is_internal_url(content, allowed_domain):
                discovered_urls.append(content)
        
        # Look for infinite scroll or lazy loading patterns
        # Check for common patterns in the HTML that suggest dynamic loading
        infinite_scroll_indicators = [
            'data-infinite-scroll',
            'data-lazy-load',
            'data-load-more',
            'infinite-scroll',
            'lazy-load',
            'load-more'
        ]
        
        for indicator in infinite_scroll_indicators:
            elements = soup.find_all(attrs={indicator: True})
            for element in elements:
                # Try to extract URLs from these elements
                for attr in ['data-url', 'data-href', 'href']:
                    url = element.get(attr)
                    if url and is_internal_url(url, allowed_domain):
                        discovered_urls.append(url)
        
        # Look for script tags that might contain blog post data
        for script in soup.find_all("script"):
            if script.string:
                # Look for common patterns in JavaScript that indicate blog posts
                patterns = [
                    r'posts\s*[:=]\s*\[([^\]]+)\]',
                    r'articles\s*[:=]\s*\[([^\]]+)\]',
                    r'blogPosts\s*[:=]\s*\[([^\]]+)\]',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, script.string, re.IGNORECASE)
                    for match in matches:
                        # Try to extract URLs from the match
                        url_matches = re.findall(r'["\']([^"\']*\/blog\/[^"\']*)["\']', match)
                        for url_match in url_matches:
                            if is_internal_url(url_match, allowed_domain):
                                discovered_urls.append(url_match)
                
    except Exception:
        pass
    
    return list(set(discovered_urls))


def discover_with_selenium(base_url: str, allowed_domain: str, max_scrolls: int = 5) -> List[str]:
    """
    Use Selenium to discover dynamically loaded content (optional dependency).
    This function will only work if selenium is installed.
    """
    discovered_urls = []
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, WebDriverException
        
        # Set up Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            driver.get(base_url)
            
            # Wait for initial page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll to trigger lazy loading
            for i in range(max_scrolls):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # Wait for content to load
                
                # Extract all links after each scroll
                links = driver.find_elements(By.TAG_NAME, "a")
                for link in links:
                    href = link.get_attribute("href")
                    if href and is_internal_url(href, allowed_domain):
                        discovered_urls.append(href)
            
            # Look for "Load More" or similar buttons and click them
            load_more_selectors = [
                "button[data-load-more]",
                "a[data-load-more]",
                ".load-more",
                ".load-more-btn",
                "[data-infinite-scroll]",
                "button:contains('Load More')",
                "a:contains('Load More')",
                "button:contains('Show More')",
                "a:contains('Show More')"
            ]
            
            for selector in load_more_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            driver.execute_script("arguments[0].click();", element)
                            time.sleep(3)  # Wait for content to load
                            
                            # Extract new links
                            links = driver.find_elements(By.TAG_NAME, "a")
                            for link in links:
                                href = link.get_attribute("href")
                                if href and is_internal_url(href, allowed_domain):
                                    discovered_urls.append(href)
                except:
                    continue
                    
        finally:
            driver.quit()
            
    except ImportError:
        print("   ℹ️ Selenium not available - skipping dynamic content discovery")
    except Exception as e:
        print(f"   ⚠️ Selenium discovery failed: {e}")
    
    return list(set(discovered_urls))


def normalize_url(base_url: str, link: str) -> str:
    return urljoin(base_url, link.split("#")[0])


def discover_sitemap_from_robots(session: requests.Session, domain: str) -> Optional[str]:
    """
    Discover sitemap URL from robots.txt file.
    Returns the first sitemap URL found, or None if no sitemap is found.
    """
    # Try both with and without www
    robots_urls = [
        f"https://{domain}/robots.txt",
        f"https://www.{domain}/robots.txt",
    ]
    
    for robots_url in robots_urls:
        try:
            resp = session.get(robots_url, timeout=10)
            if resp.status_code == 200:
                # Look for Sitemap: directive
                for line in resp.text.split('\n'):
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        print(f"   🗺️ Discovered sitemap from robots.txt: {sitemap_url}")
                        return sitemap_url
        except Exception:
            continue
    
    # Try common sitemap locations as fallback
    common_sitemap_urls = [
        f"https://{domain}/sitemap.xml",
        f"https://www.{domain}/sitemap.xml",
        f"https://{domain}/sitemap_index.xml",
        f"https://www.{domain}/sitemap_index.xml",
    ]
    
    for sitemap_url in common_sitemap_urls:
        try:
            resp = session.head(sitemap_url, timeout=5, allow_redirects=True)
            if resp.status_code == 200:
                print(f"   🗺️ Found sitemap at: {sitemap_url}")
                return sitemap_url
        except Exception:
            continue
    
    return None


def extract_all_links_from_page(session: requests.Session, url: str, allowed_domain: str, focus_prefix: Optional[str] = None) -> List[str]:
    """
    Extract all links from a page, particularly useful for blog listing pages.
    Returns a list of URLs found on the page that match the criteria.
    """
    discovered_urls = []
    
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return discovered_urls
            
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find all links
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
                
            # Normalize URL
            full_url = urljoin(url, href)
            
            # Check if internal
            if not is_internal_url(full_url, allowed_domain):
                continue
            
            # Check if matches focus prefix
            if focus_prefix and not full_url.startswith(focus_prefix):
                continue
            
            # Avoid non-article pages
            if any(pattern in full_url.lower() for pattern in [
                '/author/', '/tag/', '/category/', '/page/', '/search', '/feed', '/rss'
            ]):
                continue
            
            discovered_urls.append(full_url)
        
        # Also look for article cards with data attributes
        for card in soup.find_all(['article', 'div'], class_=lambda x: x and any(
            cls in str(x).lower() for cls in ['post', 'article', 'blog', 'card', 'entry']
        )):
            # Look for links within article cards
            for a in card.find_all("a", href=True):
                href = a.get("href")
                if href:
                    full_url = urljoin(url, href)
                    if is_internal_url(full_url, allowed_domain):
                        if not focus_prefix or full_url.startswith(focus_prefix):
                            discovered_urls.append(full_url)
        
    except Exception as e:
        print(f"   ⚠️ Error extracting links from {url}: {e}")
    
    return list(set(discovered_urls))


def normalize_url_for_comparison(url: str) -> str:
    """Normalize URL for comparison by removing www. and trailing slashes."""
    url = url.replace('://www.', '://')
    url = url.rstrip('/')
    return url


def parse_sitemap_urls(
    session: requests.Session,
    sitemap_url: str,
    allowed_domain: Optional[str] = None,
    include_prefix: Optional[str] = None,
    max_urls: int = 5000,
) -> List[str]:
    """Fetch a sitemap or sitemap index and return contained URLs.

    include_prefix: if provided, only URLs starting with this prefix are returned.
    URL matching is flexible and handles www variations.
    """
    try:
        resp = session.get(sitemap_url, timeout=20, headers={"User-Agent": "seo-graph/0.1"})
        resp.raise_for_status()
        content = resp.content
        ctype = resp.headers.get("Content-Type", "")
        if "gzip" in ctype or sitemap_url.endswith(".gz"):
            try:
                content = gzip.decompress(content)
            except Exception:
                content = BytesIO(content).read()

        # Strip namespace
        def strip_ns(tag: str) -> str:
            return tag.split("}", 1)[1] if "}" in tag else tag

        root = ET.fromstring(content)
        tag = strip_ns(root.tag)

        urls: List[str] = []
        if tag == "sitemapindex":
            children = root.findall(".//{*}sitemap") or root.findall(".//sitemap")
            for sm in children:
                loc = sm.findtext("{*}loc") or sm.findtext("loc")
                if loc:
                    urls.extend(parse_sitemap_urls(session, loc, allowed_domain, include_prefix, max_urls))
                    if len(urls) >= max_urls:
                        break
        elif tag == "urlset":
            children = root.findall(".//{*}url") or root.findall(".//url")
            # Normalize prefix for comparison if provided
            normalized_prefix = normalize_url_for_comparison(include_prefix) if include_prefix else None
            
            for u in children:
                loc = u.findtext("{*}loc") or u.findtext("loc")
                if not loc:
                    continue
                if allowed_domain and not is_internal_url(loc, allowed_domain):
                    continue
                
                # Flexible prefix matching - normalize both URLs for comparison
                if normalized_prefix:
                    normalized_loc = normalize_url_for_comparison(loc)
                    if not normalized_loc.startswith(normalized_prefix):
                        continue
                
                urls.append(loc)
                if len(urls) >= max_urls:
                    break
        else:
            return []

        # Deduplicate preserving order
        seen: Set[str] = set()
        ordered: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        return ordered
    except Exception as e:
        print(f"   ⚠️ Error parsing sitemap {sitemap_url}: {e}")
        return []


def crawl_site(
    seed_url: str,
    allowed_domain: Optional[str] = None,
    max_pages: int = 500,
    max_depth: int = 3,
    delay_seconds: float = 0.3,
    sitemap_url: Optional[str] = None,
    focus_prefix: Optional[str] = None,
    enable_js_discovery: bool = True,
    enable_pagination_discovery: bool = True,
    use_selenium: bool = False,
) -> Dict[str, Page]:
    """
    Enhanced BFS crawl with JavaScript and pagination discovery.
    Returns mapping URL -> Page.
    """
    if allowed_domain is None:
        ext = tldextract.extract(urlparse(seed_url).netloc)
        allowed_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    session = requests.Session()

    visited: Set[str] = set()
    pages: Dict[str, Page] = {}

    queue: deque[Tuple[str, int]] = deque()
    initial_urls: List[str] = [seed_url]
    
    # Auto-discover sitemap if not provided
    if sitemap_url is None:
        print(f"🔍 Auto-discovering sitemap for {allowed_domain}...")
        sitemap_url = discover_sitemap_from_robots(session, allowed_domain)
    
    # Add sitemap URLs if available
    if sitemap_url:
        print(f"📡 Fetching URLs from sitemap: {sitemap_url}")
        sm_urls = parse_sitemap_urls(session, sitemap_url, allowed_domain=allowed_domain, include_prefix=focus_prefix)
        print(f"   Found {len(sm_urls)} URLs in sitemap")
        initial_urls = sm_urls + initial_urls
    
    # Extract all links from the seed page (blog landing page)
    print(f"🔗 Extracting all links from landing page: {seed_url}")
    landing_page_urls = extract_all_links_from_page(session, seed_url, allowed_domain, focus_prefix)
    print(f"   Found {len(landing_page_urls)} links on landing page")
    initial_urls.extend(landing_page_urls)
    
    # Enhanced discovery for JavaScript-based content
    if enable_js_discovery:
        print(f"🔍 Discovering dynamic content for {seed_url}...")
        dynamic_urls = discover_dynamic_content(session, seed_url, allowed_domain)
        initial_urls.extend(dynamic_urls)
        print(f"   Found {len(dynamic_urls)} dynamic URLs")
        
        # Use Selenium for JavaScript-heavy sites if requested
        if use_selenium:
            print(f"🤖 Using Selenium for advanced JavaScript discovery...")
            selenium_urls = discover_with_selenium(seed_url, allowed_domain)
            initial_urls.extend(selenium_urls)
            print(f"   Found {len(selenium_urls)} additional URLs via Selenium")
    
    # Enhanced discovery for pagination
    if enable_pagination_discovery:
        print(f"📄 Discovering pagination for {seed_url}...")
        pagination_urls = discover_pagination_urls(session, seed_url, allowed_domain)
        initial_urls.extend(pagination_urls)
        print(f"   Found {len(pagination_urls)} pagination URLs")
    
    # Deduplicate initial URLs and filter by focus_prefix
    seen_init: Set[str] = set()
    filtered_count = 0
    normalized_focus = normalize_url_for_comparison(focus_prefix) if focus_prefix else None
    
    for u in initial_urls:
        if u not in seen_init:
            # Strictly filter by focus_prefix using normalized comparison
            if normalized_focus:
                normalized_u = normalize_url_for_comparison(u)
                if not normalized_u.startswith(normalized_focus):
                    filtered_count += 1
                    continue
            queue.append((u, 0))
            seen_init.add(u)
    
    if filtered_count > 0:
        print(f"   🔍 Filtered out {filtered_count} URLs not matching focus prefix")

    print(f"🚀 Starting crawl with {len(queue)} initial URLs...")
    
    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        
        # Strict focus_prefix check - skip URLs that don't match
        if focus_prefix:
            normalized_url = normalize_url_for_comparison(url)
            if not normalized_url.startswith(normalized_focus):
                continue
        
        visited.add(url)

        try:
            title, text, raw_links = fetch_page(session, url)
        except Exception as e:
            print(f"   ⚠️ Failed to fetch {url}: {e}")
            continue

        abs_links: List[Link] = []
        for l in raw_links:
            abs_url = normalize_url(url, l.href)
            if is_internal_url(abs_url, allowed_domain):
                abs_links.append(Link(href=abs_url, anchor=l.anchor))

        pages[url] = Page(url=url, title=title, text=text, links=abs_links, depth=depth)
        
        if len(pages) % 10 == 0:
            print(f"   📊 Crawled {len(pages)} pages so far...")

        for l in abs_links:
            if l.href not in visited and len(pages) + len(queue) < max_pages:
                # Strictly enforce focus_prefix - never crawl outside the focused section
                if focus_prefix:
                    normalized_link = normalize_url_for_comparison(l.href)
                    if not normalized_link.startswith(normalized_focus):
                        continue
                queue.append((l.href, depth + 1))

        time.sleep(delay_seconds)

    print(f"✅ Crawl completed! Found {len(pages)} pages total.")
    return pages
