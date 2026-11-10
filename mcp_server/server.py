"""MCP server exposing the Document Intelligence API as tools for any MCP client/agent.

Design: the MCP server is a thin, stateless adapter over the REST API (single source of truth,
independent scaling/auth). Tool descriptions matter: the LLM reads them to decide when to call.

Run (stdio, for Claude Desktop / MCP Inspector):   python -m mcp_server.server
Run (HTTP, for remote agents / AWS):               python -m mcp_server.server --transport streamable-http
"""

import argparse
import os
from collections.abc import Callable
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer(
    "doc-intel",
    instructions=(
        "Tools for searching and querying a private document corpus (policies, runbooks, guidelines). "
        "Prefer `search_docs` to inspect evidence, `ask_question` for a cited answer, "
        "`get_document` to read a full document, `list_documents` to see what exists."
    ),
)


def _default_client() -> httpx.AsyncClient:
    headers = {"X-API-Key": os.environ["DOC_API_KEY"]} if os.getenv("DOC_API_KEY") else {}
    return httpx.AsyncClient(base_url=os.getenv("DOC_API_URL", "http://localhost:8000"), headers=headers, timeout=30)


# Tests replace this factory to point the tools at an in-process app.
client_factory: Callable[[], httpx.AsyncClient] = _default_client


async def _call(method: str, path: str, **kwargs: Any) -> Any:
    async with client_factory() as client:
        try:
            resp = await client.request(method, path, **kwargs)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # ToolError = an *anticipated* failure. The model sees this text and can recover (e.g. fix a
            # bad doc_id). Any other exception is treated as a crash and the model only sees a generic message.
            detail = e.response.json().get("detail", e.response.text) if e.response.content else ""
            raise ToolError(f"API returned {e.response.status_code}: {detail}") from e
        except httpx.RequestError as e:
            raise ToolError(f"Cannot reach the document API: {e}") from e
        return resp.json()


@mcp.tool()
async def search_docs(query: str, top_k: int = 4) -> dict[str, Any]:
    """Semantic search over the document corpus. Returns the best-matching passages with
    doc_id, chunk_id and a similarity score (0-1). Use it to gather evidence before answering."""
    return await _call("GET", "/v1/search", params={"q": query, "k": top_k})


@mcp.tool()
async def ask_question(question: str, top_k: int = 4) -> dict[str, Any]:
    """Ask a natural-language question and get an answer grounded in the corpus, with sources
    for citation. Answers 'I don't know...' when the documents do not contain the answer."""
    return await _call("POST", "/v1/query", json={"question": question, "top_k": top_k})


@mcp.tool()
async def get_document(doc_id: str) -> dict[str, Any]:
    """Fetch the full text of one document by its doc_id (as returned by search_docs or list_documents)."""
    return await _call("GET", f"/v1/documents/{doc_id}")


@mcp.tool()
async def list_documents() -> dict[str, Any]:
    """List every document in the corpus with its size and chunk count."""
    return {"documents": await _call("GET", "/v1/documents")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Intelligence MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 inside a container)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
