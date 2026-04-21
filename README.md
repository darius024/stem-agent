# stem-agent

A minimal "stem agent" for deep research. The agent starts generic, spends a
small budget studying a training set of multi-hop questions, writes itself a
short specialization artifact, and then answers held-out questions using that
artifact. The before/after comparison is run on a fixed sample of HotpotQA.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then fill in OPENAI_API_KEY
```

Environment variables used (see `.env.example`):

- `OPENAI_API_KEY` — required.
- `OPENAI_MODEL` — pinned model id, defaults to `gpt-4o-mini`.
- `SEARCH_API_KEY` — reserved; the current tool set uses the keyless
  MediaWiki API, so this is only needed if you swap in a different backend.

## Running

The committed `data/hotpot_sample.json` is a fixed 200-question subsample of
the HotpotQA distractor dev set. `make_splits` deterministically divides it
into scout (10), self-check (30), and test (160) given the seed.

### 1. Baseline evaluation

Run the generic agent on the test split:

```bash
python scripts/run_eval.py --arm baseline --seed 0
```

### 2. Sanity floor

Run the same model with no tools at all. If this beats the agent, the
evaluation may be contaminated by memorization.

```bash
python scripts/run_eval.py --arm closed_book --seed 0
```

### 3. Growth phase

Scout on the 10 training questions, distill an artifact, and self-check it
against the 30-question internal split. The script writes candidate
artifacts into `data/artifacts/` and records which one was promoted in
`data/artifacts/promotion.json`. Rejected candidates go into
`data/artifacts/rejected/`.

```bash
python scripts/run_growth.py
```

### 4. Specialized evaluation

Point the eval runner at the promoted artifact:

```bash
python scripts/run_eval.py \
    --arm specialized \
    --artifact data/artifacts/$(jq -r .artifact_path data/artifacts/promotion.json) \
    --seed 0
```

### 5. Summary table

```bash
python scripts/summarize_results.py
```

### Regenerating the sample

The committed sample is deterministic given `--seed 42` and size 200. To
regenerate or change sizes:

```bash
python scripts/sample_hotpot.py --seed 42 --size 200
```

## Tests

```bash
pytest
```

Tests cover: metric normalization, artifact schema round-trips, tool caching
and URL routing (network stubbed), the ReAct loop with a scripted model,
dataset splits, the evaluation runner, the growth orchestration, and the
promotion/rejection logic on disk.

## Layout

```
src/stem_agent/       library code
  tools/              search and page fetch, with on-disk caching
  agent/              ReAct-style loop, model client, prompt templates
  specialization/     artifact schema, growth phase, rollback
  eval/               HotpotQA loader, metrics, evaluation runner
scripts/              entry points (sampling, growth, eval, summary)
data/                 cache, artifacts, results (gitignored contents)
tests/                small focused tests
writeup/              report
```
