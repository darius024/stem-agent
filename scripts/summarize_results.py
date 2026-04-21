"""Summarize per-question CSVs into a single table.

Example:
    python scripts/summarize_results.py
    python scripts/summarize_results.py data/results/*.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from stem_agent.config import RESULTS_DIR


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = path.name
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["seed"])].append(row)

    summaries = []
    for (arm, seed), entries in sorted(grouped.items()):
        em = [float(e["em"]) for e in entries]
        f1 = [float(e["f1"]) for e in entries]
        hops = [int(e["hops"]) for e in entries]
        elapsed = [float(e["elapsed_s"]) for e in entries]
        fallbacks = sum(1 for e in entries if e["fallback"].lower() == "true")
        errors = sum(1 for e in entries if e["error"])
        summaries.append(
            {
                "arm": arm,
                "seed": seed,
                "n": len(entries),
                "em": _mean(em),
                "f1": _mean(f1),
                "avg_hops": _mean(hops),
                "avg_s": _mean(elapsed),
                "fallbacks": fallbacks,
                "errors": errors,
            }
        )
    return summaries


def _print_table(summaries: list[dict]) -> None:
    header = ["arm", "seed", "n", "em", "f1", "avg_hops", "avg_s", "fallbacks", "errors"]
    print("{:<14} {:>4} {:>4} {:>6} {:>6} {:>9} {:>7} {:>10} {:>7}".format(*header))
    for s in summaries:
        print(
            "{:<14} {:>4} {:>4} {:>6.3f} {:>6.3f} {:>9.2f} {:>7.2f} {:>10d} {:>7d}".format(
                s["arm"],
                s["seed"],
                s["n"],
                s["em"],
                s["f1"],
                s["avg_hops"],
                s["avg_s"],
                s["fallbacks"],
                s["errors"],
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="CSV files to summarize.")
    args = parser.parse_args()

    paths = args.paths or sorted(RESULTS_DIR.glob("*.csv"))
    if not paths:
        print("no CSV files found", file=sys.stderr)
        return 1

    rows = _load_rows(paths)
    summaries = _summarize(rows)
    _print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
