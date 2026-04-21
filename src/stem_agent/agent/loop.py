"""ReAct-style agent loop.

The loop is intentionally small. It alternates between the model producing
tool calls and the caller executing those tools, until the model calls the
special ``answer`` tool or the hop cap is reached.

Only one thing differs between the baseline and the specialized agent: which
system prompt is passed in. The loop, tools, and model are shared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from stem_agent.agent.model import ModelClient, ModelResponse, ToolCall


TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia for pages related to the query. "
                "Use short, specific queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Fetch the cleaned text of a URL, truncated.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Submit the final short answer and stop.",
            "parameters": {
                "type": "object",
                "properties": {"final": {"type": "string"}},
                "required": ["final"],
            },
        },
    },
]


@dataclass
class LoopResult:
    answer: str
    hops: int
    fallback: bool
    error: str | None = None
    trace: list[dict] = field(default_factory=list)


def run_loop(
    question: str,
    system_prompt: str,
    model: ModelClient,
    tools: dict[str, Callable[..., object]],
    max_hops: int = 8,
) -> LoopResult:
    """Run the ReAct loop until `answer` is called or `max_hops` is reached."""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    fallback = False

    for hop in range(max_hops):
        response = model.complete(messages, TOOL_SPEC)

        if not response.tool_calls:
            # Model gave a plain text reply; treat it as the final answer.
            return LoopResult(
                answer=(response.content or "").strip(),
                hops=hop,
                fallback=fallback,
                trace=trace,
            )

        messages.append(_assistant_message(response))

        for call in response.tool_calls:
            trace.append({"hop": hop, "tool": call.name, "args": call.arguments})

            if call.name == "answer":
                return LoopResult(
                    answer=str(call.arguments.get("final", "")).strip(),
                    hops=hop + 1,
                    fallback=fallback,
                    trace=trace,
                )

            observation, had_error = _dispatch_tool(tools, call)
            if had_error:
                fallback = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                }
            )

    return LoopResult(
        answer="",
        hops=max_hops,
        fallback=fallback,
        error="hop_cap_exceeded",
        trace=trace,
    )


def _assistant_message(response: ModelResponse) -> dict:
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in response.tool_calls
        ],
    }


def _dispatch_tool(
    tools: dict[str, Callable[..., object]], call: ToolCall
) -> tuple[str, bool]:
    function = tools.get(call.name)
    if function is None:
        return f"unknown tool: {call.name}", True
    try:
        result = function(**call.arguments)
    except TypeError as exc:
        return f"bad arguments to {call.name}: {exc}", True
    except Exception as exc:  # noqa: BLE001 — network/parse errors are fed back to the model
        return f"{call.name} failed: {exc}", True
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return result, False
