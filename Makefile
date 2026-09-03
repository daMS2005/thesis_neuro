# Development shortcuts that mirror the CI quality gates.

PYTHON ?= python
CODE_PATHS = src scripts tests

.PHONY: install install-full lint compile test notebooks links hygiene smoke check

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-full:
	$(PYTHON) -m pip install -e ".[full]"

lint:
	ruff check $(CODE_PATHS)

compile:
	$(PYTHON) -m compileall -q $(CODE_PATHS)

test:
	pytest

notebooks:
	$(PYTHON) scripts/quality/validate_notebooks.py

links:
	$(PYTHON) scripts/quality/check_markdown_links.py

hygiene:
	$(PYTHON) scripts/quality/check_repo_hygiene.py

smoke:
	thesis-neuro --config configs/examples/mock.yaml mock-extract
	thesis-neuro-benchmark validate-items --items-path examples/benchmark/mock_boolq.jsonl

check: lint compile test notebooks links hygiene smoke
