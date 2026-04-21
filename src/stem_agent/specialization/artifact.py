"""Specialization artifact: schema, load, save.

The artifact is the only thing that differs between the baseline and the
specialized agent. Keeping it as a plain JSON file means runs are easy to
diff and rollback is just picking an older file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Artifact:
    query_style_rules: list[str] = field(default_factory=list)
    source_preferences: dict = field(default_factory=lambda: {"prefer": [], "avoid": []})
    stopping_heuristic: str = ""
    typical_hop_count: int = 3
    few_shots: list[dict] = field(default_factory=list)
    system_prompt: str = ""


REQUIRED_KEYS = {
    "query_style_rules",
    "source_preferences",
    "stopping_heuristic",
    "typical_hop_count",
    "few_shots",
    "system_prompt",
}


def save(artifact: Artifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(artifact), indent=2, ensure_ascii=False))


def load(path: Path) -> Artifact:
    raw = json.loads(path.read_text())
    missing = REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"artifact at {path} missing keys: {sorted(missing)}")
    return Artifact(**{k: raw[k] for k in REQUIRED_KEYS})
