"""Exercises the MCP server through a real MCP client, with tools wired to an in-process API."""

import httpx
import pytest
from mcp.client.client import Client

import mcp_server.server as srv
from app.main import create_app


@pytest.fixture
async def wired(settings, monkeypatch):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        monkeypatch.setattr(
            srv,
            "client_factory",
            lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"),
        )
        yield


def _payload(result):
    assert not result.is_error, result
    return result.structured_content


async def test_tools_are_listed_with_descriptions(wired):
    async with Client(srv.mcp) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"search_docs", "ask_question", "get_document", "list_documents"}
    assert all(t.description for t in tools.values())


async def test_search_docs_tool(wired):
    async with Client(srv.mcp) as c:
        out = _payload(await c.call_tool("search_docs", {"query": "tier 1 vendor reassessment", "top_k": 2}))
    assert out["results"][0]["doc_id"] == "vendor_risk_policy"


async def test_ask_question_tool(wired):
    async with Client(srv.mcp) as c:
        out = _payload(await c.call_tool("ask_question", {"question": "How is the QA compliance dimension weighted?"}))
    assert out["sources"][0]["doc_id"] == "call_quality_guidelines"


async def test_missing_document_returns_readable_error(wired):
    async with Client(srv.mcp) as c:
        result = await c.call_tool("get_document", {"doc_id": "ghost"})
    assert result.is_error
    assert "404" in result.content[0].text
