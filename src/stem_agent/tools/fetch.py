"""Page fetch tool.

For Wikipedia URLs we ask the MediaWiki "extracts" endpoint for a clean
plain-text body, which avoids running an HTML parser on the huge rendered
article. For anything else we fall back to a simple httpx + BeautifulSoup
extraction. Responses are cached on disk keyed by URL.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from stem_agent.config import CACHE_DIR, load_settings

WIKI_API = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = 20.0


def _cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / "fetch" / f"{digest}.txt"


def _wiki_title(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.netloc.endswith("wikipedia.org"):
        return None
    prefix = "/wiki/"
    if not parsed.path.startswith(prefix):
        return None
    return unquote(parsed.path[len(prefix) :]).replace("_", " ")


def _fetch_wiki_extract(title: str) -> str:
    response = httpx.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages:
        return ""
    return pages[0].get("extract", "") or ""


def _fetch_html(url: str) -> str:
    response = httpx.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def open_url(url: str) -> str:
    """Return cleaned page text for `url`, truncated to the page char limit."""
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text()

    title = _wiki_title(url)
    text = _fetch_wiki_extract(title) if title else _fetch_html(url)
    text = text[: load_settings().page_char_limit]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text
