export UV_PROJECT_ENVIRONMENT ?= .venv

setup:
	uv sync --group dev

check:
	uv run ruff check . && uv run ruff format --check . && uv run mypy perception scripts
