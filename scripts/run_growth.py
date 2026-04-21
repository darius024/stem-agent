"""Run the growth phase end-to-end and save the produced artifact(s).

Example:
    python scripts/run_growth.py
    python scripts/run_growth.py --scout-size 5 --self-check-size 10

The script prints a short report and writes:
- ``data/artifacts/artifact_<timestamp>.json`` for each candidate
- ``data/artifacts/promotion.json`` pointing at the live artifact
- ``data/artifacts/rejected/...`` for candidates that lost the comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stem_agent.agent.model import OpenAIModel
from stem_agent.config import ARTIFACTS_DIR, DATA_DIR, load_settings
from stem_agent.eval.dataset import load_sample, make_splits
from stem_agent.specialization.grow import grow
from stem_agent.specialization.rollback import save_and_promote, save_as_rejected

DEFAULT_SAMPLE = DATA_DIR / "hotpot_sample.json"
DISTILL_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "stem_agent" / "agent" / "prompts" / "distill.txt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=0, help="Split seed.")
    parser.add_argument("--scout-size", type=int, default=10)
    parser.add_argument("--self-check-size", type=int, default=30)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--max-revisions", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings()
    questions = load_sample(args.sample)
    splits = make_splits(
        questions,
        scout_size=args.scout_size,
        self_check_size=args.self_check_size,
        seed=args.seed,
    )
    print(
        f"growth: scout={len(splits.scout)} self_check={len(splits.self_check)} "
        f"(test held out: {len(splits.test)})",
        file=sys.stderr,
    )

    model = OpenAIModel(settings.openai_api_key, settings.openai_model)
    distill_prompt = DISTILL_PROMPT_PATH.read_text()

    candidates, report = grow(
        scout_questions=splits.scout,
        self_check_questions=splits.self_check,
        model=model,
        distill_prompt=distill_prompt,
        max_hops=args.max_hops,
        max_revisions=args.max_revisions,
        logger=lambda m: print(m, file=sys.stderr),
    )

    # Save every candidate; promote the best one if it beat baseline by the margin.
    saved_paths = []
    for index, (candidate, score) in enumerate(
        zip(candidates, report.candidate_self_check_f1s)
    ):
        if report.promoted_index == index:
            path = save_and_promote(ARTIFACTS_DIR, candidate, self_check_f1=score)
        else:
            path = save_as_rejected(ARTIFACTS_DIR, candidate)
        saved_paths.append(path)

    summary = {
        "baseline_self_check_f1": round(report.baseline_self_check_f1, 4),
        "candidate_self_check_f1s": [round(f, 4) for f in report.candidate_self_check_f1s],
        "promoted_index": report.promoted_index,
        "artifact_paths": [str(p) for p in saved_paths],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
