# CLAUDE.md

This file provides guidance to coding agents (Claude Code, and Codex through `AGENTS.md`, a symlink
to this file) when working with code in this repository.

## What this is

The code and experiments behind a paper on **knowledge domains in language-model corpora**: graph
indicators, computed on a corpus alone, of whether it holds the patterns a model needs to
generalize. Extracted from `eva-llm` (commit `00d8312` plus its uncommitted analysis changes of
2026-09-27), where it was `make tokenizer-analyze-corpus`; it has no dependency on that project.

## Commands

Requires Python `>=3.13,<3.14` and uv (in-project `.venv`; `uv sync` installs, `uv.lock` pins).

- `make check` — lint + format-check + typecheck + test. Run this before finishing any change.
- `make analyze CORPUS=<name>` — `corpus-analyze --corpus <name>` (console script,
  `corpus_generalization.analyze:main`): reads `configs/corpora/<name>/tokenizer_analysis.json`,
  writes `analysis/<name>/<date>/` (gitignored). `analyze.py`'s module docstring documents every
  config key and every output file.
- `corpus-analyze --config <path> [--output <dir>]` — the same analysis on any config file, from
  any project: `files` and the default `analysis/` resolve against the working directory. eva-llm
  depends on this package (path dependency) and runs its own models' configs this way.
- `corpus-diagnose --files <paths>` / `make diagnose ARGS="--files ..."` — diagnose per-pattern
  syntactic sufficiency, optionally against a reference; writes JSON and Markdown reports. See
  `docs/superpowers/specs/2026-09-30-corpus-diagnose-design.md`.
- `make train CONFIG=<path>` — `corpus-train --config <path>` (`corpus_generalization.model.train`):
  trains the fixed GPT on a corpus, on the CPU (never MPS: measured far slower), into
  `runs/<config folder>/<date>/` (gitignored). `corpus-generate --run <dir> --prompt <text>` samples
  from a run; `make ui` opens the Streamlit page (`ui/app.py`) that lists every run and generates.
- Single test: `uv run pytest tests/test_markov_chain_analysis.py::test_name`
- `experiments/<name>/` — one folder per experiment: its script, its results JSON and a README
  with the finding. Run with `uv run python experiments/<name>/<script>.py`.
- Specs go to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, with a `**Status:**` line
  under the title (draft, approved, running, done, with the date and where the result is). Plans
  go to `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`, with `- [ ]` steps checked off as they
  run. Other notes (`docs/2026-10-05-related-work.md`) stay in `docs/`.

## Architecture

- `src/corpus_generalization/analyze.py` — the command: config → report.
- `src/corpus_generalization/diagnose.py` — syntactic diagnosis pipeline, reports, and command.
- `src/corpus_generalization/analysis/` — the metrics. `markov_chain.py` holds the transition
  graph (`build_transition_graph`) and the four coverage curves with their linear and log AUC
  (its module docstring is the full derivation); `markov_graph.py` draws them; `tokenization.py`,
  `branching.py`, `kl_divergence.py`, `coverage.py`, `spectral.py`, `domains.py` are the other
  report sections; `neo4j_export.py` pushes the graph to Neo4j.
  `syntactic.py` is the syntactic-sufficiency indicator (coarse classes, head → class patterns,
  coverage), `docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md`.
- `src/corpus_generalization/tokenizer/` — byte-level BPE (`bpe.py`) and its fixed-capacity
  variant (`reserved_bpe.py`). Token ids 0–4 are the five specials of `base.py`, bytes are 5–260,
  merges follow: kept identical to `eva-llm` so both produce the same ids, and
  `tests/test_bpe_tokenizer.py` pins that against `tests/fixtures/bpe_training_texts.json`.
- `src/corpus_generalization/model/` — the calibration model: `gpt.py` (GPT-2-layout decoder,
  tied head), `data.py` (document-level train/held-out split, framed token streams), `train.py`
  and `generate.py` (the two commands); `scoring.py` reads a continuation's log probability.
  `ui/app.py` is the Streamlit page over `runs/`.
- `src/corpus_generalization/synthetic/` — generated corpora with known patterns: `recipes.py`,
  the recipe grammar, its variants and minimal pairs.
- `sources.py` (files / Hugging Face), `packing.py` (`<bos>`/`<eos>` framing), `corpora.py`
  (corpus names → config folders), `config_file.py` (schema-checked JSON configs).
- `data/kid_learning/` — the hand-built corpora, committed: they are the paper's evidence.
- `tests/` is the executable specification; `tests/conftest.py` points `configs/corpora/` at an
  empty temp dir for every test.

## Style

- Strict typing (mypy `strict = true`) and Ruff (line length 100); use `pathlib`.
- Comments record *why* (a measured reason for a design choice), not what the code does.
- A metric's behavior is what the paper reports: change one only with a test that pins the new
  behavior, and re-run the affected experiments.
- Don't commit or branch unless explicitly asked.
