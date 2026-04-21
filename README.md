# stem-agent

A minimal "stem agent" for deep research. The agent starts generic, spends a
small budget studying a training set of multi-hop questions, writes itself a
short specialization artifact, and then answers held-out questions using that
artifact. The before/after comparison is run on a fixed sample of HotpotQA.

This is a student project built as a submission for the JetBrains AI
Engineering internship (Task #1: Stem Agent).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then fill in your keys
```

Environment variables used (see `.env.example`):

- `OPENAI_API_KEY` — required.
- `OPENAI_MODEL` — pinned model id, defaults to `gpt-4o-mini`.
- `SEARCH_API_KEY` — search backend key (Tavily or similar). Optional until
  the tool is wired up.

## Running

Nothing is runnable yet. The scaffold is in place; scripts under `scripts/`
will be added as the project grows. See `writeup/writeup.md` for the planned
approach.

## Tests

```bash
pytest
```

## Layout

```
src/stem_agent/       library code
  tools/              search and page fetch, with on-disk caching
  agent/              ReAct-style loop and prompt templates
  specialization/     artifact schema, growth phase, rollback
  eval/               HotpotQA loader, metrics, evaluation runner
scripts/              entry points (sampling, growth, eval, summary)
data/                 cache, artifacts, results (gitignored contents)
tests/                small focused tests
writeup/              report
```
