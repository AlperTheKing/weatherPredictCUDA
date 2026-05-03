.PHONY: fix lint test test-fast check

fix:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run --extra opendata ty check src tests

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

check: lint test
