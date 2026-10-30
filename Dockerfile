FROM python:3.14.6-slim-bookworm@sha256:4c92ffcde4dd6f1ff72a24518f49fd4990b27134987dfa31a733badde66df9f8 AS base
RUN pip install --no-cache-dir --disable-pip-version-check uv==0.11.29
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --uid 10001 --create-home app && mkdir /app/artifacts && chown app:app /app/artifacts
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --python /usr/local/bin/python

FROM base AS production
COPY --chown=app:app scripts ./scripts
USER app
EXPOSE 8000
CMD ["uv", "run", "--locked", "--no-dev", "--no-sync", "uvicorn", "dialogentis.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM base AS test
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --python /usr/local/bin/python
COPY tests ./tests
COPY scripts ./scripts
RUN chown app:app /app && chown -R app:app /app/tests /app/scripts
USER app
CMD ["uv", "run", "--locked", "--no-sync", "pytest", "-q"]
