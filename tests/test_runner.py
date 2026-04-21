"""Test the evaluation runner with a scripted model and fake tools."""

from __future__ import annotations

from pathlib import Path

from stem_agent.agent.model import ModelResponse, ScriptedModel, ToolCall
from stem_agent.eval.dataset import Question
from stem_agent.eval.runner import (
    baseline_runner,
    closed_book_runner,
    compose_specialized_prompt,
    evaluate,
    specialized_runner,
    write_results_csv,
)
from stem_agent.specialization.artifact import Artifact


def _answer_call(final: str) -> ToolCall:
    return ToolCall(id="a", name="answer", arguments={"final": final})


def _fake_tools():
    return {
        "search": lambda query: [{"title": "t", "url": "u", "snippet": "s"}],
        "open_url": lambda url: "content",
    }


def test_baseline_runner_scores_each_question() -> None:
    model = ScriptedModel(
        [
            ModelResponse(None, [_answer_call("Paris")]),
            ModelResponse(None, [_answer_call("wrong")]),
        ]
    )
    questions = [
        Question(id="q1", question="Capital of France?", answer="Paris"),
        Question(id="q2", question="Capital of Germany?", answer="Berlin"),
    ]
    run_fn = baseline_runner(model, max_hops=3, tools=_fake_tools())
    results = evaluate(questions, arm="baseline", seed=0, run_fn=run_fn)

    assert [r.em for r in results] == [1.0, 0.0]
    assert [r.prediction for r in results] == ["Paris", "wrong"]


def test_closed_book_runner_uses_content_directly() -> None:
    model = ScriptedModel([ModelResponse(content="Berlin", tool_calls=[])])
    questions = [Question(id="q1", question="Capital of Germany?", answer="Berlin")]
    run_fn = closed_book_runner(model)
    results = evaluate(questions, arm="closed_book", seed=0, run_fn=run_fn)

    assert results[0].em == 1.0
    assert results[0].hops == 0


def test_specialized_prompt_contains_artifact_parts() -> None:
    artifact = Artifact(
        query_style_rules=["be specific"],
        source_preferences={"prefer": ["wikipedia.org"], "avoid": []},
        stopping_heuristic="two sources agree",
        typical_hop_count=3,
        few_shots=[{"question": "q?", "trajectory": "t", "answer": "a"}],
        system_prompt="You are a focused research agent.",
    )
    prompt = compose_specialized_prompt(artifact)

    assert "focused research agent" in prompt
    assert "be specific" in prompt
    assert "wikipedia.org" in prompt
    assert "two sources agree" in prompt
    assert "Example Q: q?" in prompt


def test_specialized_runner_runs_loop_with_artifact_prompt() -> None:
    artifact = Artifact(system_prompt="custom prompt")
    model = ScriptedModel([ModelResponse(None, [_answer_call("yes")])])
    questions = [Question(id="q1", question="anything?", answer="yes")]
    run_fn = specialized_runner(model, artifact, max_hops=3, tools=_fake_tools())
    results = evaluate(questions, arm="specialized", seed=0, run_fn=run_fn)

    assert results[0].em == 1.0


def test_csv_writer_round_trips_fields(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(None, [_answer_call("Paris")])])
    questions = [Question(id="q1", question="Capital?", answer="Paris")]
    run_fn = baseline_runner(model, max_hops=1, tools=_fake_tools())
    results = evaluate(questions, arm="baseline", seed=0, run_fn=run_fn)

    path = tmp_path / "out.csv"
    write_results_csv(path, results)
    lines = path.read_text().splitlines()

    assert lines[0].split(",")[:3] == ["id", "arm", "seed"]
    assert "Paris" in lines[1]
