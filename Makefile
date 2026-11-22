.PHONY: sync check test integration acceptance up
sync:
	uv sync --locked
check:
	uv run --locked ruff format --check .
	uv run --locked ruff check .
	uv run --locked ty check
	uv run --locked python scripts/export_openapi.py --check
test:
	uv run --locked pytest -m 'not integration'
integration:
	sh scripts/integration.sh
acceptance:
	@: "$${CHROME_PATH:?Set CHROME_PATH to the Chrome or Chromium executable}"
	$(MAKE) check test integration
	sh scripts/browser_acceptance.sh
up:
	docker compose -f compose.yaml -f compose.dev.yaml --profile dev up --build -d
