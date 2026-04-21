"""Run an evaluation arm over a list of questions and write per-question CSV.

Three arms are supported:
- ``baseline``: generic system prompt, full tool set, loop with a hop cap.
- ``specialized``: artifact-derived system prompt, same tools, same cap.
- ``closed_book``: one model call with no tools (sanity floor).

The only thing that differs between baseline and specialized at run time is
the composed system prompt. The loop, tools, and model are identical.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from stem_agent.agent.loop import LoopResult, run_loop
from stem_agent.agent.model import ModelClient
from stem_agent.eval.dataset import Question
from stem_agent.eval.metrics import exact_match, token_f1
from stem_agent.specialization.artifact import Artifact
from stem_agent.tools.fetch import open_url
from stem_agent.tools.search import search

PROMPT_DIR = Path(__file__).resolve().parent.parent / "agent" / "prompts"


@dataclass
class QuestionResult:
    id: str
    arm: str
    seed: int
    prediction: str
    gold: str
    em: float
    f1: float
    hops: int
    fallback: bool
    error: str | None
    elapsed_s: float


# ---------------------------------------------------------------------------
# Arm builders. Each returns a callable Question -> LoopResult.


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def _default_tools() -> dict[str, Callable[..., object]]:
    return {"search": search, "open_url": open_url}


def baseline_runner(
    model: ModelClient,
    max_hops: int = 8,
    tools: dict | None = None,
) -> Callable[[Question], LoopResult]:
    system_prompt = _read_prompt("baseline.txt")
    bound_tools = tools if tools is not None else _default_tools()

    def _run(question: Question) -> LoopResult:
        return run_loop(question.question, system_prompt, model, bound_tools, max_hops)

    return _run


def specialized_runner(
    model: ModelClient,
    artifact: Artifact,
    max_hops: int = 8,
    tools: dict | None = None,
) -> Callable[[Question], LoopResult]:
    system_prompt = compose_specialized_prompt(artifact)
    bound_tools = tools if tools is not None else _default_tools()

    def _run(question: Question) -> LoopResult:
        return run_loop(question.question, system_prompt, model, bound_tools, max_hops)

    return _run


def closed_book_runner(model: ModelClient) -> Callable[[Question], LoopResult]:
    system_prompt = (
        "Answer the question briefly and directly, in as few words as possible. "
        "You have no tools; rely only on what you know."
    )

    def _run(question: Question) -> LoopResult:
        response = model.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question.question},
            ],
            tools=[],
        )
        return LoopResult(answer=(response.content or "").strip(), hops=0, fallback=False)

    return _run


def compose_specialized_prompt(artifact: Artifact) -> str:
    """Stitch the artifact into a single system prompt. Order is deterministic."""
    parts: list[str] = []
    if artifact.system_prompt.strip():
        parts.append(artifact.system_prompt.strip())
    else:
        parts.append(_read_prompt("baseline.txt").strip())

    if artifact.query_style_rules:
        bullet = "\n- ".join(artifact.query_style_rules)
        parts.append(f"Query style rules:\n- {bullet}")

    prefer = artifact.source_preferences.get("prefer", [])
    avoid = artifact.source_preferences.get("avoid", [])
    if prefer or avoid:
        parts.append(f"Prefer sources: {prefer}. Avoid: {avoid}.")

    if artifact.stopping_heuristic:
        parts.append(f"Stopping heuristic: {artifact.stopping_heuristic}")

    if artifact.few_shots:
        shots = "\n\n".join(
            f"Example Q: {shot.get('question','')}\n"
            f"Trajectory: {shot.get('trajectory','')}\n"
            f"A: {shot.get('answer','')}"
            for shot in artifact.few_shots
        )
        parts.append(f"Examples:\n{shots}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Evaluation driver.


def evaluate(
    questions: list[Question],
    arm: str,
    seed: int,
    run_fn: Callable[[Question], LoopResult],
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for question in questions:
        started = time.perf_counter()
        try:
            loop_result = run_fn(question)
            error = loop_result.error
        except Exception as exc:  # noqa: BLE001 — one question failure must not kill the run
            loop_result = LoopResult(answer="", hops=0, fallback=True)
            error = f"uncaught:{type(exc).__name__}:{exc}"
        elapsed = time.perf_counter() - started

        results.append(
            QuestionResult(
                id=question.id,
                arm=arm,
                seed=seed,
                prediction=loop_result.answer,
                gold=question.answer,
                em=exact_match(loop_result.answer, question.answer),
                f1=token_f1(loop_result.answer, question.answer),
                hops=loop_result.hops,
                fallback=loop_result.fallback,
                error=error,
                elapsed_s=round(elapsed, 3),
            )
        )
    return results


CSV_FIELDS = [
    "id",
    "arm",
    "seed",
    "prediction",
    "gold",
    "em",
    "f1",
    "hops",
    "fallback",
    "error",
    "elapsed_s",
]


def write_results_csv(path: Path, results: list[QuestionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
