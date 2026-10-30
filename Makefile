.PHONY: sync check test integration up
sync:
	uv sync --locked
check:
	uv run --locked ruff format --check .
	uv run --locked ruff check .
	uv run --locked ty check
test:
	uv run --locked pytest -m 'not integration'
integration:
	sh scripts/integration.sh
up:
	docker compose -f compose.yaml -f compose.dev.yaml --profile dev up --build -d
