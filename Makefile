.PHONY: install test run mcp-inspector smoke docker lint format
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
test:
	python -m pytest -q
run:
	uvicorn app.main:create_app --factory --reload --port 8000
smoke:               # needs `make run` in another terminal
	python scripts/mcp_smoke.py
mcp-inspector:       # opens a browser UI to call your MCP tools by hand
	npx @modelcontextprotocol/inspector python -m mcp_server.server
docker:
	docker compose up --build

lint:                # exactly what CI runs
	ruff check . && ruff format --check .
format:
	ruff format . && ruff check --fix .
