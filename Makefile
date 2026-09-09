.PHONY: setup demo test check

setup:
	uv sync --locked

demo:
	uv run --locked sdcf-v4 run --fundamentals examples/synthetic_fundamentals.csv --snapshot examples/synthetic_snapshot.yaml --output output/example

test:
	uv run --locked pytest

check:
	uv run --locked ruff format --check .
	uv run --locked ruff check .
	uv run --locked mypy
	uv run --locked pytest
