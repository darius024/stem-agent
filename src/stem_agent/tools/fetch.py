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

from bs4 import BeautifulSoup

from stem_agent.config import CACHE_DIR, load_settings
from stem_agent.tools._http import get_json, get_text

WIKI_API = "https://en.wikipedia.org/w/api.php"


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
    payload = get_json(
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
    )
    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return ""
    return pages[0].get("extract", "") or ""


def _fetch_html(url: str) -> str:
    html = get_text(url)
    soup = BeautifulSoup(html, "html.parser")
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
