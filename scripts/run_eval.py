"""Run one evaluation arm on the held-out test split and write a CSV.

Example:
    python scripts/run_eval.py --arm baseline --limit 5 --seed 0
    python scripts/run_eval.py --arm specialized --artifact data/artifacts/live.json
    python scripts/run_eval.py --arm closed_book
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stem_agent.agent.model import OpenAIModel
from stem_agent.config import DATA_DIR, RESULTS_DIR, load_settings
from stem_agent.eval.dataset import load_sample, make_splits
from stem_agent.eval.runner import (
    baseline_runner,
    closed_book_runner,
    evaluate,
    specialized_runner,
    write_results_csv,
)
from stem_agent.specialization.artifact import load as load_artifact

DEFAULT_SAMPLE = DATA_DIR / "hotpot_sample.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["baseline", "specialized", "closed_book"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--artifact", type=Path, help="Required for --arm specialized.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N test questions.")
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--output", type=Path, help="CSV output path.")
    args = parser.parse_args()

    settings = load_settings()
    questions = load_sample(args.sample)
    splits = make_splits(questions, seed=args.seed)
    test = splits.test if args.limit is None else splits.test[: args.limit]
    print(f"evaluating {len(test)} questions (arm={args.arm}, seed={args.seed})", file=sys.stderr)

    model = OpenAIModel(settings.openai_api_key, settings.openai_model)

    if args.arm == "baseline":
        run_fn = baseline_runner(model, max_hops=args.max_hops)
    elif args.arm == "closed_book":
        run_fn = closed_book_runner(model)
    else:
        if args.artifact is None:
            parser.error("--artifact is required for --arm specialized")
        artifact = load_artifact(args.artifact)
        run_fn = specialized_runner(model, artifact, max_hops=args.max_hops)

    results = evaluate(test, arm=args.arm, seed=args.seed, run_fn=run_fn)

    output = args.output or RESULTS_DIR / f"{args.arm}_seed{args.seed}.csv"
    write_results_csv(output, results)

    em = sum(r.em for r in results) / len(results)
    f1 = sum(r.f1 for r in results) / len(results)
    print(
        f"done: n={len(results)} EM={em:.3f} F1={f1:.3f} -> {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
