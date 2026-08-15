FROM ghcr.io/astral-sh/uv:debian-slim AS uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app
ENV UV_NO_DEV=1
# COPY pyproject.toml uv.lock ./
RUN uv sync --locked

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
