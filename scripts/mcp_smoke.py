"""Spawns the MCP server over stdio (exactly how Claude Desktop / the Inspector do) and calls each tool.
Usage: python scripts/mcp_smoke.py   (API must be running on DOC_API_URL, default http://localhost:8000)"""

import asyncio
import sys

from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])
    async with Client(params) as c:
        tools = (await c.list_tools()).tools
        print("tools:", [t.name for t in tools])
        r = await c.call_tool("ask_question", {"question": "How are compliance failures scored in call QA?"})
        print("ask_question ->", r.structured_content["answer"])
        r = await c.call_tool("search_docs", {"query": "incident severity 1 communication", "top_k": 2})
        print("search_docs  ->", [(x["doc_id"], x["score"]) for x in r.structured_content["results"]])
        r = await c.call_tool("get_document", {"doc_id": "ghost"})
        print("get_document(ghost) is_error:", r.is_error, "|", r.content[0].text)


asyncio.run(main())
