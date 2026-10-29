FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy AGENT_DATA_DIR=/data FRONTEND_DIR=/frontend
COPY pyproject.toml uv.lock .python-version ./
COPY rasa/pyproject.toml rasa/uv.lock rasa/.python-version ./rasa/
RUN uv sync --locked --no-dev && uv sync --project rasa --locked
COPY app ./app
COPY demo.py ./
RUN useradd --create-home atlas && mkdir /data /frontend && chown atlas:atlas /data /frontend
USER atlas
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
