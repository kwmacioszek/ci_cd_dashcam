export UV_PROJECT_ENVIRONMENT ?= .venv
UI_ARGS ?=

setup:
	uv sync --extra ml --group dev

check:
	uv run ruff check . && uv run ruff format --check . && uv run mypy perception scripts

ui:
	# Użyj aktywnego środowiska ROCm bez synchronizacji do wersji CPU.
	uv run --active --no-sync python -m perception.app $(UI_ARGS)

cov:
	uv run pytest -m "not model" --cov --cov-report=term-missing --cov-report=html -q
