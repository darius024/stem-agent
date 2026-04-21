"""HotpotQA sample loader and split helpers.

The committed sample file is a list of objects with the fields we actually
need: `id`, `question`, `answer`, `level`. Splits are deterministic given
the seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    type: str = ""


@dataclass(frozen=True)
class Splits:
    scout: list[Question]
    self_check: list[Question]
    test: list[Question]


def load_sample(path: Path) -> list[Question]:
    raw = json.loads(path.read_text())
    return [
        Question(
            id=entry["id"],
            question=entry["question"],
            answer=entry["answer"],
            type=entry.get("type", ""),
        )
        for entry in raw
    ]


def make_splits(
    questions: list[Question],
    scout_size: int = 10,
    self_check_size: int = 30,
    seed: int = 0,
) -> Splits:
    if scout_size + self_check_size >= len(questions):
        raise ValueError("scout + self_check must leave room for a test split")
    rng = random.Random(seed)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    scout = shuffled[:scout_size]
    self_check = shuffled[scout_size : scout_size + self_check_size]
    test = shuffled[scout_size + self_check_size :]
    return Splits(scout=scout, self_check=self_check, test=test)
