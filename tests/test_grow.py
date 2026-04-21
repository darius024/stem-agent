"""Tests for the growth phase: distill parsing and grow() orchestration."""

from __future__ import annotations

import json

import pytest

from stem_agent.agent.model import ModelResponse, ScriptedModel, ToolCall
from stem_agent.eval.dataset import Question
from stem_agent.specialization.artifact import Artifact
from stem_agent.specialization.grow import (
    _extract_json_object,
    _parse_artifact,
    distill,
    grow,
)


# ---------------------------------------------------------------------------
# JSON parsing is where models misbehave; cover the common shapes.


_VALID_ARTIFACT = {
    "query_style_rules": ["use entity names"],
    "source_preferences": {"prefer": ["wikipedia.org"], "avoid": []},
    "stopping_heuristic": "stop when the answer appears in two pages",
    "typical_hop_count": 3,
    "few_shots": [],
    "system_prompt": "You are specialized.",
}


def test_extract_json_object_handles_plain_json() -> None:
    raw = json.dumps(_VALID_ARTIFACT)
    assert _extract_json_object(raw) == _VALID_ARTIFACT


def test_extract_json_object_strips_markdown_fence() -> None:
    raw = f"```json\n{json.dumps(_VALID_ARTIFACT)}\n```"
    assert _extract_json_object(raw) == _VALID_ARTIFACT


def test_extract_json_object_handles_prose_around_json() -> None:
    raw = f"Here you go:\n{json.dumps(_VALID_ARTIFACT)}\nThanks."
    assert _extract_json_object(raw) == _VALID_ARTIFACT


def test_parse_artifact_rejects_missing_keys() -> None:
    incomplete = dict(_VALID_ARTIFACT)
    incomplete.pop("stopping_heuristic")
    with pytest.raises(ValueError):
        _parse_artifact(json.dumps(incomplete))


def test_distill_returns_artifact_from_scripted_model() -> None:
    model = ScriptedModel(
        [ModelResponse(content=json.dumps(_VALID_ARTIFACT), tool_calls=[])]
    )
    artifact = distill(trajectories=[], model=model, distill_prompt="distill")

    assert isinstance(artifact, Artifact)
    assert artifact.stopping_heuristic == _VALID_ARTIFACT["stopping_heuristic"]


# ---------------------------------------------------------------------------
# grow() orchestration: scout -> distill -> self-check -> (maybe) revise.


def _answer(final: str) -> ToolCall:
    return ToolCall(id="c", name="answer", arguments={"final": final})


def _fake_tools():
    return {"search": lambda query: [], "open_url": lambda url: ""}


def test_grow_promotes_when_candidate_beats_baseline(monkeypatch) -> None:
    scout_q = [Question(id="s1", question="q?", answer="yes")]
    check_q = [
        Question(id="c1", question="q?", answer="yes"),
        Question(id="c2", question="q?", answer="yes"),
    ]

    # Scout answers (1) -> baseline on self-check (2 wrong) ->
    # distill produces artifact -> specialized on self-check (2 correct).
    responses = [
        ModelResponse(None, [_answer("yes")]),                       # scout s1
        ModelResponse(None, [_answer("no")]),                        # baseline c1
        ModelResponse(None, [_answer("no")]),                        # baseline c2
        ModelResponse(content=json.dumps(_VALID_ARTIFACT), tool_calls=[]),  # distill
        ModelResponse(None, [_answer("yes")]),                       # specialized c1
        ModelResponse(None, [_answer("yes")]),                       # specialized c2
    ]
    model = ScriptedModel(responses)

    candidates, report = grow(
        scout_questions=scout_q,
        self_check_questions=check_q,
        model=model,
        distill_prompt="distill",
        tools=_fake_tools(),
        logger=lambda msg: None,
    )

    assert len(candidates) == 1
    assert report.promoted_index == 0
    assert report.baseline_self_check_f1 == 0.0
    assert report.candidate_self_check_f1s == [1.0]


def test_grow_attempts_revision_when_first_candidate_fails(monkeypatch) -> None:
    scout_q = [Question(id="s1", question="q?", answer="yes")]
    check_q = [Question(id="c1", question="q?", answer="yes")]

    responses = [
        ModelResponse(None, [_answer("yes")]),                              # scout
        ModelResponse(None, [_answer("yes")]),                              # baseline c1 = perfect
        ModelResponse(content=json.dumps(_VALID_ARTIFACT), tool_calls=[]),  # distill v1
        ModelResponse(None, [_answer("no")]),                               # specialized v1 = wrong
        ModelResponse(content=json.dumps(_VALID_ARTIFACT), tool_calls=[]),  # distill v2
        ModelResponse(None, [_answer("no")]),                               # specialized v2 = wrong
    ]
    model = ScriptedModel(responses)

    candidates, report = grow(
        scout_questions=scout_q,
        self_check_questions=check_q,
        model=model,
        distill_prompt="distill",
        max_revisions=1,
        tools=_fake_tools(),
        logger=lambda msg: None,
    )

    assert len(candidates) == 2
    assert report.promoted_index is None
    assert report.baseline_self_check_f1 == 1.0
