FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv

# zależności osobno: ta warstwa przebudowuje się tylko po zmianie uv.lock/pyproject
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra ml --extra ui

COPY perception ./perception
RUN uv sync --frozen --no-editable --extra ml --extra ui

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" HF_HOME=/cache/huggingface
RUN useradd -m app && mkdir -p /cache/huggingface && chown -R app /cache
USER app
EXPOSE 7860
ENTRYPOINT ["perception-ui", "--host", "0.0.0.0", "--port", "7860"]
