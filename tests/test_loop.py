"""Test the agent loop end-to-end with a scripted model and fake tools."""

from __future__ import annotations

from stem_agent.agent.loop import run_loop
from stem_agent.agent.model import ModelResponse, ScriptedModel, ToolCall


def _call(id_: str, name: str, **args) -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args)


def _fake_tools():
    return {
        "search": lambda query: [{"title": "X", "url": "u", "snippet": "s"}],
        "open_url": lambda url: "Paris is the capital of France.",
    }


def test_loop_runs_search_then_open_then_answer() -> None:
    model = ScriptedModel(
        [
            ModelResponse(None, [_call("a", "search", query="capital of France")]),
            ModelResponse(None, [_call("b", "open_url", url="https://x")]),
            ModelResponse(None, [_call("c", "answer", final="Paris")]),
        ]
    )
    result = run_loop("What is the capital of France?", "sys", model, _fake_tools())

    assert result.answer == "Paris"
    assert result.hops == 3
    assert result.fallback is False
    assert [step["tool"] for step in result.trace] == ["search", "open_url", "answer"]


def test_loop_flags_fallback_on_unknown_tool_and_continues() -> None:
    model = ScriptedModel(
        [
            ModelResponse(None, [_call("a", "not_a_tool")]),
            ModelResponse(None, [_call("b", "answer", final="ok")]),
        ]
    )
    result = run_loop("q", "sys", model, _fake_tools())

    assert result.answer == "ok"
    assert result.fallback is True


def test_loop_reports_hop_cap_when_model_never_answers() -> None:
    model = ScriptedModel(
        [ModelResponse(None, [_call(f"id{i}", "search", query="x")]) for i in range(4)]
    )
    result = run_loop("q", "sys", model, _fake_tools(), max_hops=3)

    assert result.error == "hop_cap_exceeded"
    assert result.answer == ""
    assert result.hops == 3


def test_loop_treats_plain_text_as_final_answer() -> None:
    model = ScriptedModel([ModelResponse(content="Paris", tool_calls=[])])
    result = run_loop("q", "sys", model, _fake_tools())

    assert result.answer == "Paris"
    assert result.hops == 0


def test_loop_reports_fallback_when_tool_raises() -> None:
    def boom(**_):
        raise RuntimeError("network down")

    model = ScriptedModel(
        [
            ModelResponse(None, [_call("a", "search", query="x")]),
            ModelResponse(None, [_call("b", "answer", final="done")]),
        ]
    )
    result = run_loop("q", "sys", model, {"search": boom, "open_url": lambda url: ""})

    assert result.answer == "done"
    assert result.fallback is True
