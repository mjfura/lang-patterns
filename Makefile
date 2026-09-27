# corpus-generalization — common commands. Run `make` for this list.

POETRY := poetry
RUN    := $(POETRY) run

# Which configs/corpora/<name>/tokenizer_analysis.json to analyze (CORPUS=analysis-17-wikipedia-en).
CORPUS      ?=
ARGS        ?=
CORPUS_FLAG := $(if $(CORPUS),--corpus $(CORPUS))

.DEFAULT_GOAL := help
.PHONY: help install test lint format format-check typecheck check clean analyze

help:  ## Show this list
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies into the in-project venv
	$(POETRY) install

test:  ## Run the test suite
	$(RUN) pytest

lint:  ## Lint with ruff
	$(RUN) ruff check .

format:  ## Reformat in place with ruff
	$(RUN) ruff format .

format-check:  ## Check formatting without writing
	$(RUN) ruff format --check .

typecheck:  ## Type-check src and tests with mypy
	$(RUN) mypy

check: lint format-check typecheck test  ## Every gate, in the order a failure is cheapest to read

analyze:  ## Tokenization metrics, coverage curves and domain structure of a corpus (CORPUS=name)
	$(RUN) python scripts/tokenizer_analyze_corpus.py $(CORPUS_FLAG) $(ARGS)

clean:  ## Remove tool caches and bytecode
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
