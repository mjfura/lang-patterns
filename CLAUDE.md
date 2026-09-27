# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The code and experiments behind a paper on **knowledge domains in language-model corpora**: graph
indicators, computed on a corpus alone, of whether it holds the patterns a model needs to
generalize. Extracted from `eva-llm` (commit `00d8312` plus its uncommitted analysis changes of
2026-09-27), where it was `make tokenizer-analyze-corpus`; it has no dependency on that project.

## Commands

Requires Python `>=3.13,<3.14` and Poetry (in-project venv, `poetry.toml`).

- `make check` — lint + format-check + typecheck + test. Run this before finishing any change.
- `make analyze CORPUS=<name>` — `scripts/tokenizer_analyze_corpus.py --corpus <name>`: reads
  `configs/corpora/<name>/tokenizer_analysis.json`, writes `analysis/<name>/<date>/` (gitignored).
  The script's module docstring documents every config key and every output file.
- Single test: `poetry run pytest tests/test_markov_chain_analysis.py::test_name`
- `experiments/<name>/` — one folder per experiment: its script, its results JSON and a README
  with the finding. Run with `poetry run python experiments/<name>/<script>.py`.

## Architecture

- `src/corpus_generalization/analysis/` — the metrics. `markov_chain.py` holds the transition
  graph (`build_transition_graph`) and the four coverage curves with their linear and log AUC
  (its module docstring is the full derivation); `markov_graph.py` draws them; `tokenization.py`,
  `branching.py`, `kl_divergence.py`, `coverage.py`, `spectral.py`, `domains.py` are the other
  report sections; `neo4j_export.py` pushes the graph to Neo4j.
- `src/corpus_generalization/tokenizer/` — byte-level BPE (`bpe.py`) and its fixed-capacity
  variant (`reserved_bpe.py`). Token ids 0–4 are the five specials of `base.py`, bytes are 5–260,
  merges follow: kept identical to `eva-llm` so both produce the same ids, and
  `tests/test_bpe_tokenizer.py` pins that against `tests/fixtures/bpe_training_texts.json`.
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
