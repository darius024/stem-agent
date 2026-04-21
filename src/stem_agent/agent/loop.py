"""ReAct-style agent loop. Not implemented yet.

The loop will keep its state as a plain list of (thought, action, observation)
triples. The only difference between the baseline and the specialized agent
is which system prompt and few-shots are loaded; the loop itself is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoopState:
    question: str
    scratchpad: list[dict] = field(default_factory=list)
    hops_used: int = 0
    final_answer: str | None = None


def run_loop(state: LoopState, system_prompt: str) -> LoopState:
    raise NotImplementedError
