# Development shortcuts that mirror the CI quality gates.

PYTHON ?= python
LINT_PATHS = src benchmark_comparison structure_comparison eda_brain_data/scripts tests scripts

.PHONY: install install-full lint compile test notebooks links hygiene smoke check

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-full:
	$(PYTHON) -m pip install -e ".[full]"

lint:
	ruff check $(LINT_PATHS)

compile:
	$(PYTHON) -m compileall -q $(LINT_PATHS)

test:
	pytest

notebooks:
	$(PYTHON) scripts/validate_notebooks.py

links:
	$(PYTHON) scripts/check_markdown_links.py

hygiene:
	$(PYTHON) scripts/check_repo_hygiene.py

smoke:
	thesis-neuro --config configs/examples/mock.yaml mock-extract
	thesis-neuro-benchmark validate-items --items-path examples/benchmark/mock_boolq.jsonl

check: lint compile test notebooks links hygiene smoke
