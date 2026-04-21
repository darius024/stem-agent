"""Page fetch tool. Not implemented yet.

Planned behavior: `open_url(url)` returns cleaned page text truncated to
`Settings.page_char_limit`. Results are cached on disk keyed by URL.
"""

from __future__ import annotations


def open_url(url: str) -> str:
    raise NotImplementedError
