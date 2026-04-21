"""Growth phase: scout -> distill -> self-check -> promote/reject.

The point of this module is to turn a small set of example questions into a
specialization artifact, and to decide whether that artifact is actually
better than the baseline on a held-out internal split.

Nothing here touches the final test split. That is only read by the eval
runner after growth is frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

from stem_agent.agent.loop import run_loop
from stem_agent.agent.model import ModelClient
from stem_agent.eval.dataset import Question
from stem_agent.eval.metrics import token_f1
from stem_agent.eval.runner import baseline_runner, evaluate, specialized_runner
from stem_agent.specialization.artifact import REQUIRED_KEYS, Artifact


@dataclass
class ScoutTrajectory:
    question: str
    gold: str
    prediction: str
    f1: float
    trace: list[dict]


@dataclass
class GrowthReport:
    baseline_self_check_f1: float
    candidate_self_check_f1s: list[float]
    promoted_index: int | None  # index into candidate list, or None if all rejected
    trajectories: list[ScoutTrajectory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scout: run the baseline on the scout split and record trajectories.


def run_scout(
    questions: list[Question],
    model: ModelClient,
    max_hops: int = 8,
    tools: dict | None = None,
) -> list[ScoutTrajectory]:
    run_fn = baseline_runner(model, max_hops=max_hops, tools=tools)
    trajectories: list[ScoutTrajectory] = []
    for question in questions:
        result = run_fn(question)
        trajectories.append(
            ScoutTrajectory(
                question=question.question,
                gold=question.answer,
                prediction=result.answer,
                f1=token_f1(result.answer, question.answer),
                trace=result.trace,
            )
        )
    return trajectories


# ---------------------------------------------------------------------------
# Distill: turn trajectories into an artifact.


def distill(
    trajectories: list[ScoutTrajectory],
    model: ModelClient,
    distill_prompt: str,
    previous_artifact: Artifact | None = None,
) -> Artifact:
    user_content = {
        "trajectories": [asdict(t) for t in trajectories],
    }
    if previous_artifact is not None:
        user_content["previous_artifact"] = asdict(previous_artifact)
        user_content["revision_note"] = (
            "The previous artifact was not accepted. Produce an improved version "
            "that addresses the failures visible in the trajectories."
        )

    response = model.complete(
        [
            {"role": "system", "content": distill_prompt},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
        tools=[],
    )
    return _parse_artifact(response.content or "")


def _parse_artifact(raw: str) -> Artifact:
    payload = _extract_json_object(raw)
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise ValueError(f"distilled artifact missing keys: {sorted(missing)}")
    return Artifact(**{key: payload[key] for key in REQUIRED_KEYS})


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json") :]
        text = text.strip("` \n")
    # Sometimes models add prose around the JSON. Fall back to slicing on braces.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Scoring on the internal self-check split.


def score_baseline(
    questions: list[Question],
    model: ModelClient,
    max_hops: int = 8,
    tools: dict | None = None,
) -> float:
    run_fn = baseline_runner(model, max_hops=max_hops, tools=tools)
    results = evaluate(questions, arm="baseline_selfcheck", seed=0, run_fn=run_fn)
    return mean(r.f1 for r in results)


def score_specialized(
    artifact: Artifact,
    questions: list[Question],
    model: ModelClient,
    max_hops: int = 8,
    tools: dict | None = None,
) -> float:
    run_fn = specialized_runner(model, artifact, max_hops=max_hops, tools=tools)
    results = evaluate(questions, arm="specialized_selfcheck", seed=0, run_fn=run_fn)
    return mean(r.f1 for r in results)


# ---------------------------------------------------------------------------
# Top-level orchestration.


PROMOTION_MARGIN = 0.03  # the artifact must beat baseline by this absolute F1 margin


def grow(
    scout_questions: list[Question],
    self_check_questions: list[Question],
    model: ModelClient,
    distill_prompt: str,
    *,
    max_hops: int = 8,
    max_revisions: int = 1,
    tools: dict | None = None,
    logger: Callable[[str], None] = print,
) -> tuple[list[Artifact], GrowthReport]:
    """Run the growth phase. Returns (candidates, report).

    The caller decides what to do with the promoted artifact (save, compare to
    the current promotion, etc.). This function does not write to disk.
    """
    logger("scout: running baseline on scout split")
    trajectories = run_scout(scout_questions, model, max_hops, tools)
    logger(
        f"scout: mean F1 on scout split = "
        f"{mean(t.f1 for t in trajectories):.3f}"
    )

    logger("self-check: scoring baseline on self-check split")
    baseline_f1 = score_baseline(self_check_questions, model, max_hops, tools)
    logger(f"self-check: baseline F1 = {baseline_f1:.3f}")

    candidates: list[Artifact] = []
    candidate_scores: list[float] = []
    promoted_index: int | None = None
    previous_artifact: Artifact | None = None

    for attempt in range(max_revisions + 1):
        label = "initial" if attempt == 0 else f"revision {attempt}"
        logger(f"distill: producing {label} artifact")
        candidate = distill(trajectories, model, distill_prompt, previous_artifact)
        candidates.append(candidate)

        logger(f"self-check: scoring {label} artifact")
        candidate_f1 = score_specialized(candidate, self_check_questions, model, max_hops, tools)
        candidate_scores.append(candidate_f1)
        logger(
            f"self-check: {label} F1 = {candidate_f1:.3f} "
            f"(baseline {baseline_f1:.3f}, margin needed {PROMOTION_MARGIN})"
        )

        if candidate_f1 >= baseline_f1 + PROMOTION_MARGIN:
            promoted_index = attempt
            break
        previous_artifact = candidate

    return candidates, GrowthReport(
        baseline_self_check_f1=baseline_f1,
        candidate_self_check_f1s=candidate_scores,
        promoted_index=promoted_index,
        trajectories=trajectories,
    )
