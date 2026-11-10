"""A tiny OpenAI-compatible server (embeddings + chat, incl. streaming) for testing the 'openai'
provider path with no API key and no network:

    python scripts/fake_openai_server.py            # listens on :9000
    EMBEDDING_PROVIDER=openai LLM_PROVIDER=openai OPENAI_API_KEY=x \
      OPENAI_BASE_URL=http://localhost:9000/v1 uvicorn app.main:create_app --factory
"""

import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from app.embeddings import HashEmbedder

app = FastAPI()
_emb = HashEmbedder(384)


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
    vecs = await _emb.embed(inputs)
    return {
        "object": "list",
        "model": body["model"],
        "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    user = body["messages"][-1]["content"]
    has_system = body["messages"][0]["role"] == "system"
    text = f"FAKE-LLM saw {len(user)} chars of prompt; system prompt present: {has_system}. [1]"
    base = {"id": "chatcmpl-fake", "created": int(time.time()), "model": body["model"]}
    if not body.get("stream"):
        return {
            **base,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def gen():
        for word in text.split(" "):
            chunk = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="warning")
