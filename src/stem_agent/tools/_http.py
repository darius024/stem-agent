"""Small helpers shared by the tool modules."""

from __future__ import annotations

import time

import httpx

USER_AGENT = "stem-agent/0.0.1 (research prototype; contact via repository)"
DEFAULT_TIMEOUT = 20.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 0.5


def get_json(url: str, params: dict) -> dict:
    return _get(url, params=params).json()


def get_text(url: str) -> str:
    return _get(url).text


def _get(url: str, params: dict | None = None) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = httpx.get(
                url,
                params=params,
                timeout=DEFAULT_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            # Retry only on transient 429/5xx statuses.
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"status {response.status_code}", request=response.request, response=response
                )
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(BACKOFF_SECONDS * (2**attempt))
    assert last_exc is not None
    raise last_exc
