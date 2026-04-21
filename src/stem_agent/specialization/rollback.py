"""Artifact promotion and rollback.

Specialization artifacts are versioned by timestamp in the artifacts
directory. A tiny ``promotion.json`` records which artifact is "live" and
what self-check F1 it was promoted with. Rejected candidates are saved
directly under ``rejected/`` so that the history is visible without shadowing
the live artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from stem_agent.specialization.artifact import Artifact, save

PROMOTION_FILE = "promotion.json"
REJECTED_DIR = "rejected"


@dataclass(frozen=True)
class Promotion:
    artifact_path: str  # stored as POSIX path relative to the artifacts root
    self_check_f1: float


def _timestamped_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{prefix}_{stamp}.json"


def _fresh_path(directory: Path, prefix: str) -> Path:
    while True:
        candidate = directory / _timestamped_name(prefix)
        if not candidate.exists():
            return candidate


def current_promotion(artifacts_dir: Path) -> Promotion | None:
    path = artifacts_dir / PROMOTION_FILE
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return Promotion(artifact_path=raw["artifact_path"], self_check_f1=raw["self_check_f1"])


def save_and_promote(artifacts_dir: Path, artifact: Artifact, self_check_f1: float) -> Path:
    """Save `artifact` as the live one and update ``promotion.json``."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = _fresh_path(artifacts_dir, "artifact")
    save(artifact, path)
    promotion = Promotion(
        artifact_path=path.relative_to(artifacts_dir).as_posix(),
        self_check_f1=self_check_f1,
    )
    (artifacts_dir / PROMOTION_FILE).write_text(
        json.dumps(
            {"artifact_path": promotion.artifact_path, "self_check_f1": promotion.self_check_f1},
            indent=2,
        )
    )
    return path


def save_as_rejected(artifacts_dir: Path, artifact: Artifact) -> Path:
    """Save `artifact` directly under ``rejected/`` without touching the live link."""
    rejected_dir = artifacts_dir / REJECTED_DIR
    rejected_dir.mkdir(parents=True, exist_ok=True)
    path = _fresh_path(rejected_dir, "artifact")
    save(artifact, path)
    return path
