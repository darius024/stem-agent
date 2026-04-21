"""Tests for tool caching and URL routing. All network calls are stubbed."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stem_agent.tools import fetch, search


@pytest.fixture
def _redirect_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("stem_agent.tools.search.CACHE_DIR", tmp_path)
    monkeypatch.setattr("stem_agent.tools.fetch.CACHE_DIR", tmp_path)
    return tmp_path


def test_search_returns_cleaned_results_and_caches(_redirect_cache, monkeypatch):
    payload = {
        "query": {
            "search": [
                {"title": "Barack Obama", "snippet": "44th <b>president</b>"},
                {"title": "Obama Foundation", "snippet": "founded in 2014"},
            ]
        }
    }
    fake_get = MagicMock(return_value=payload)
    monkeypatch.setattr("stem_agent.tools.search.get_json", fake_get)

    first = search.search("who is obama")
    second = search.search("who is obama")

    assert fake_get.call_count == 1  # second call was served from cache
    assert first == second
    assert first[0]["url"] == "https://en.wikipedia.org/wiki/Barack_Obama"
    assert "<b>" not in first[0]["snippet"]


def test_open_url_uses_extracts_endpoint_for_wikipedia(_redirect_cache, monkeypatch):
    payload = {"query": {"pages": [{"extract": "Paris is the capital of France."}]}}
    fake_get = MagicMock(return_value=payload)
    monkeypatch.setattr("stem_agent.tools.fetch.get_json", fake_get)

    text_a = fetch.open_url("https://en.wikipedia.org/wiki/Paris")
    text_b = fetch.open_url("https://en.wikipedia.org/wiki/Paris")

    assert fake_get.call_count == 1
    assert text_a == text_b == "Paris is the capital of France."
    assert fake_get.call_args.kwargs["params"]["titles"] == "Paris"


def test_open_url_falls_back_to_html_for_non_wikipedia(_redirect_cache, monkeypatch):
    html = "<html><body><p>Hello</p><script>alert('x')</script> world </body></html>"
    fake_get = MagicMock(return_value=html)
    monkeypatch.setattr("stem_agent.tools.fetch.get_text", fake_get)

    text = fetch.open_url("https://example.com/post")

    assert "Hello" in text
    assert "alert" not in text
