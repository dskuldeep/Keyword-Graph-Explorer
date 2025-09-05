from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
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
    resp = session.get(url, timeout=timeout, headers={"User-Agent": "seo-graph/0.1"})
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


def normalize_url(base_url: str, link: str) -> str:
    return urljoin(base_url, link.split("#")[0])


def parse_sitemap_urls(
    session: requests.Session,
    sitemap_url: str,
    allowed_domain: Optional[str] = None,
    include_prefix: Optional[str] = None,
    max_urls: int = 5000,
) -> List[str]:
    """Fetch a sitemap or sitemap index and return contained URLs.

    include_prefix: if provided, only URLs starting with this prefix are returned.
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
            for u in children:
                loc = u.findtext("{*}loc") or u.findtext("loc")
                if not loc:
                    continue
                if allowed_domain and not is_internal_url(loc, allowed_domain):
                    continue
                if include_prefix and not loc.startswith(include_prefix):
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
    except Exception:
        return []


def crawl_site(
    seed_url: str,
    allowed_domain: Optional[str] = None,
    max_pages: int = 500,
    max_depth: int = 3,
    delay_seconds: float = 0.3,
    sitemap_url: Optional[str] = None,
    focus_prefix: Optional[str] = None,
) -> Dict[str, Page]:
    """
    BFS crawl limited to the same registered domain. Returns mapping URL -> Page.
    """
    if allowed_domain is None:
        ext = tldextract.extract(urlparse(seed_url).netloc)
        allowed_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    session = requests.Session()

    visited: Set[str] = set()
    pages: Dict[str, Page] = {}

    queue: deque[Tuple[str, int]] = deque()
    initial_urls: List[str] = [seed_url]
    if sitemap_url:
        sm_urls = parse_sitemap_urls(session, sitemap_url, allowed_domain=allowed_domain, include_prefix=focus_prefix)
        initial_urls = sm_urls + initial_urls
    # Deduplicate initial URLs
    seen_init: Set[str] = set()
    for u in initial_urls:
        if u not in seen_init:
            queue.append((u, 0))
            seen_init.add(u)

    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        try:
            title, text, raw_links = fetch_page(session, url)
        except Exception:
            continue

        abs_links: List[Link] = []
        for l in raw_links:
            abs_url = normalize_url(url, l.href)
            if is_internal_url(abs_url, allowed_domain):
                abs_links.append(Link(href=abs_url, anchor=l.anchor))

        pages[url] = Page(url=url, title=title, text=text, links=abs_links, depth=depth)

        for l in abs_links:
            if l.href not in visited and len(pages) + len(queue) < max_pages:
                # Allow one-hop expansion into non-focused URLs, but do not expand further
                if focus_prefix and not l.href.startswith(focus_prefix) and depth >= 1:
                    continue
                queue.append((l.href, depth + 1))

        time.sleep(delay_seconds)

    return pages
