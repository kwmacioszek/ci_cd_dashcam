export UV_PROJECT_ENVIRONMENT ?= .venv
UI_ARGS ?=

setup:
	uv sync --extra ml --group dev

check:
	uv run ruff check . && uv run ruff format --check . && uv run mypy perception

ui:
	# Użyj aktywnego środowiska ROCm bez synchronizacji do wersji CPU.
	uv run --active --no-sync python -m perception.app $(UI_ARGS)

cov:
	uv run pytest -m "not model" --cov --cov-report=term-missing --cov-report=html -q

docker:  ## buduje obraz z UI Gradio
	docker build -t perception:dev .

docker-run: docker  ## uruchamia kontener z UI Gradio
	docker run --rm -p 7860:7860 -v hf-cache:/cache/huggingface perception:dev

val-data:
	python scripts/fetch_val.py --split valid

eval-base:
	uv run --extra ml python scripts/eval_base.py --json runs/base_val.json
