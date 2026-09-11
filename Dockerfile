# Multi-stage build: dependencies resolve in a layer that only changes when the
# lockfile does, so source edits rebuild in seconds.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras

FROM base AS runtime
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --all-extras

# Run as an unprivileged user. The container has no reason to write outside
# its own working directory.
RUN useradd --create-home --uid 1000 arbiter && chown -R arbiter:arbiter /app
USER arbiter

ENTRYPOINT ["arbiter"]
CMD ["--help"]
