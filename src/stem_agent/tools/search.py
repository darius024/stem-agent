"""Search tool backed by the MediaWiki API.

We pick Wikipedia deliberately: HotpotQA was constructed from Wikipedia, and
using a free, stable, keyless API keeps the evaluation reproducible. Results
are cached on disk keyed by the query string so reruns stay deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from stem_agent.config import CACHE_DIR
from stem_agent.tools._http import get_json

WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARCH_LIMIT = 5


def _cache_path(query: str) -> Path:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / "search" / f"{digest}.json"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def search(query: str) -> list[dict]:
    """Return up to SEARCH_LIMIT Wikipedia hits for `query`.

    Each hit is ``{"title": ..., "url": ..., "snippet": ...}``. Cached to disk.
    """
    cache = _cache_path(query)
    if cache.exists():
        return json.loads(cache.read_text())

    payload = get_json(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": SEARCH_LIMIT,
            "format": "json",
            "formatversion": 2,
        },
    )
    hits = payload.get("query", {}).get("search", [])
    results = [
        {
            "title": hit["title"],
            "url": f"https://en.wikipedia.org/wiki/{hit['title'].replace(' ', '_')}",
            "snippet": _strip_html(hit.get("snippet", "")),
        }
        for hit in hits
    ]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results
