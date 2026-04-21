"""Download the HotpotQA distractor dev set and write a fixed subsample.

The output JSON is committed to the repo so evaluations are reproducible
without re-downloading ~46 MB each run. Only the fields we use are kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import httpx

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "hotpot_sample.json"
DEFAULT_SIZE = 200
DEFAULT_SEED = 42


def _download(url: str) -> list[dict]:
    print(f"downloading {url}", file=sys.stderr)
    response = httpx.get(url, timeout=180.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def _sample(data: list[dict], size: int, seed: int) -> list[dict]:
    pool = [entry for entry in data if entry.get("question") and entry.get("answer")]
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} valid entries, cannot sample {size}")
    chosen = random.Random(seed).sample(pool, size)
    return [
        {
            "id": entry["_id"],
            "question": entry["question"],
            "answer": entry["answer"],
            "type": entry.get("type", ""),
        }
        for entry in chosen
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Local copy of the HotpotQA dev distractor JSON (skips download).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    data = json.loads(args.input.read_text()) if args.input else _download(HOTPOT_URL)
    chosen = _sample(data, args.size, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chosen, indent=2, ensure_ascii=False))
    print(f"wrote {len(chosen)} questions to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
