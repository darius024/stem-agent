"""Central configuration for the stem agent.

Values here are read once at import time. Keep this file small: anything that
changes between runs should live in a script argument or an artifact, not
here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
REJECTED_DIR = ARTIFACTS_DIR / "rejected"
RESULTS_DIR = DATA_DIR / "results"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    search_api_key: str
    max_hops: int = 8
    page_char_limit: int = 4000


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        search_api_key=os.environ.get("SEARCH_API_KEY", ""),
    )
