"""Search tool. Not implemented yet.

Planned behavior: one function `search(query)` that returns a list of
{title, url, snippet} dicts. Results are cached to disk keyed by the query
string so that reruns of the evaluation are deterministic.
"""

from __future__ import annotations


def search(query: str) -> list[dict]:
    raise NotImplementedError
