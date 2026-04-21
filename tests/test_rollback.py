"""Tests for artifact promotion and rejection on disk."""

from __future__ import annotations

from pathlib import Path

from stem_agent.specialization.artifact import Artifact, load
from stem_agent.specialization.rollback import (
    REJECTED_DIR,
    current_promotion,
    save_and_promote,
    save_as_rejected,
)


def _artifact(prompt: str) -> Artifact:
    return Artifact(system_prompt=prompt)


def test_save_and_promote_updates_live_pointer(tmp_path: Path) -> None:
    path = save_and_promote(tmp_path, _artifact("v1"), self_check_f1=0.5)

    live = current_promotion(tmp_path)
    assert live is not None
    assert live.self_check_f1 == 0.5
    assert (tmp_path / live.artifact_path) == path
    assert load(path).system_prompt == "v1"


def test_promoting_again_overwrites_the_live_pointer(tmp_path: Path) -> None:
    save_and_promote(tmp_path, _artifact("v1"), self_check_f1=0.5)
    path_v2 = save_and_promote(tmp_path, _artifact("v2"), self_check_f1=0.6)

    live = current_promotion(tmp_path)
    assert live.self_check_f1 == 0.6
    assert (tmp_path / live.artifact_path) == path_v2


def test_save_as_rejected_stores_under_rejected_subdir(tmp_path: Path) -> None:
    save_and_promote(tmp_path, _artifact("v1"), self_check_f1=0.7)
    rejected_path = save_as_rejected(tmp_path, _artifact("v_bad"))

    assert rejected_path.parent.name == REJECTED_DIR
    assert load(rejected_path).system_prompt == "v_bad"

    # Live artifact was not touched.
    live = current_promotion(tmp_path)
    assert live.self_check_f1 == 0.7
