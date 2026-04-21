"""Model client abstraction.

The loop only needs two things from the model: produce either a final text
answer or a list of tool calls. Keeping this behind a narrow interface means
tests can drive the loop with a ScriptedModel instead of the real API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall]


class ModelClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...


class OpenAIModel:
    """Thin wrapper around the OpenAI Chat Completions API with tool calling."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0) -> None:
        # Imported lazily so tests that do not need OpenAI do not need the key.
        from openai import OpenAI

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            )
        return ModelResponse(content=message.content, tool_calls=calls)


class ScriptedModel:
    """Returns preset responses in order. For tests only."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        if not self._responses:
            raise AssertionError("ScriptedModel ran out of responses")
        self.calls += 1
        return self._responses.pop(0)
