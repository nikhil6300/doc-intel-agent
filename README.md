# Document Intelligence Agent: Week 1

A production-shaped RAG service plus an MCP server, built as the foundation for the 8-week plan.

```
            ┌────────────── MCP clients (Claude Desktop, ADK / OpenAI Agents SDK agents, Inspector) ──┐
            │  stdio or streamable-http                                                               │
            ▼                                                                                         │
   mcp_server/server.py  ──HTTP──▶  app/main.py (FastAPI)  ──▶  app/rag.py (ingest → retrieve → generate)
   tools: search_docs, ask_question,        │ /v1/query, /v1/query/stream (SSE),        │
          get_document, list_documents      │ /v1/search, /v1/ingest, /v1/documents     ├─ chunking.py
                                            │ /health, /ready                           ├─ embeddings.py (hash | openai-compatible)
                                            ▼                                           ├─ vectorstore.py (FAISS, persisted)
                                       Pydantic schemas                                 └─ llm.py (extractive | openai-compatible)
```

## Quickstart (offline, no API keys)

```bash
make install && . .venv/bin/activate        # or: python -m venv .venv && pip install -r requirements-dev.txt
cp .env.example .env
make test                                   # 28 tests
make run                                    # http://localhost:8000/docs  (interactive Swagger UI)
```

```bash
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/v1/query -H 'content-type: application/json' \
     -d '{"question":"How often must Tier 1 vendors be reassessed?"}' | python -m json.tool
curl -sN -X POST localhost:8000/v1/query/stream -H 'content-type: application/json' \
     -d '{"question":"What triggers a coaching session?"}'          # Server-Sent Events
```

## MCP server

```bash
make mcp-inspector            # browser UI: call each tool by hand (needs Node + `make run` running)
make smoke                    # scripted client over stdio, like Claude Desktop does
python -m mcp_server.server --transport streamable-http --host 0.0.0.0 --port 8080   # remote-capable
```
Env vars: `DOC_API_URL` (default `http://localhost:8000`), `DOC_API_KEY` (if the API has `API_KEY` set).

## Docker

```bash
docker compose up --build     # api on :8000, MCP (HTTP) on :8080/mcp
```
(Written and reviewed, but not executed in the environment this was authored in. If the build fails, that is your first debugging exercise.)

## Real models

Set in `.env`:
```
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=...      # optional: any OpenAI-compatible endpoint, e.g. your LiteLLM gateway
```
`python scripts/fake_openai_server.py` is a local fake provider for testing this path without a key.

## Honest limitations (so you know what Week 2+ fixes)

- Offline `hash` embeddings are **lexical, not semantic**, and the `extractive` LLM picks sentences by word overlap. They exist so the plumbing is testable. Judge answer *quality* only with real models, and only with evals (Week 2).
- FAISS flat index lives in process memory: run **one worker per container**. Week 2 moves to pgvector.
- Only a shared API key for auth. No per-user auth, rate limiting, or PII handling yet.
- No tracing or metrics yet (Week 3).

## Week 1 "done when" checklist

- [ ] `make test` is green and you can explain what each test protects.
- [ ] You called every MCP tool from the Inspector, and watched one fail on purpose (bad `doc_id`).
- [ ] You swapped to real embeddings and compared `/v1/search` scores to the hash embedder.
- [ ] You ingested 20+ real public documents (not client data) via `/v1/ingest`.
- [ ] You wrote **30 question/expected-document pairs** in `data/golden.jsonl`. This is the Week 2 eval set.
- [ ] `docker compose up` works on your machine.

## Stretch exercises

1. PDF ingestion endpoint (`UploadFile` + `pypdf`), with a size limit.
2. Rate limit `/v1/query` per API key.
3. Return `usage`/token counts from the LLM and surface them in the response.
4. Add a `/v1/query` timeout and a test that proves the 502 path.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | every PR, push to `main` | `lint` (ruff) and `test` (pytest) in parallel, then `docker` (build the image and smoke-test the running container) |
| `.github/workflows/cd.yml` | push a tag like `v0.1.0` | reruns CI, publishes the image to `ghcr.io/<owner>/doc-intel-agent`, then a `deploy` job that waits for manual approval (placeholder until Week 7) |
| `.github/dependabot.yml` | weekly | PRs for outdated pip, Actions, and Docker dependencies |

Run the same checks locally before pushing: `make lint && make test`.
