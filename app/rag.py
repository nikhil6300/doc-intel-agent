"""RAG orchestration: ingest -> retrieve -> generate. No FastAPI imports here, so it is unit-testable
and reusable from scripts, the MCP server, and (Week 2) the eval harness."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app.chunking import chunk_text
from app.config import Settings
from app.embeddings import Embedder
from app.llm import LLM, NO_ANSWER
from app.schemas import Chunk, Document, QueryResponse, Source
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """The embedding/LLM provider failed. Mapped to HTTP 502 by the API layer."""


class RAGService:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, llm: LLM) -> None:
        self.settings, self.embedder, self.store, self.llm = settings, embedder, store, llm

    # ---- ingestion ---------------------------------------------------
    async def ingest(self, docs: list[Document]) -> tuple[int, int]:
        total_chunks = 0
        for doc in docs:
            texts = chunk_text(doc.text, self.settings.chunk_size, self.settings.chunk_overlap)
            if not texts:
                continue
            chunks = [
                Chunk(chunk_id=f"{doc.doc_id}::{i}", doc_id=doc.doc_id, index=i, text=t) for i, t in enumerate(texts)
            ]
            try:
                vectors = await self.embedder.embed(texts)
            except Exception as e:  # noqa: BLE001
                raise UpstreamError(f"embedding failed: {e}") from e
            await asyncio.to_thread(self.store.add_document, doc, chunks, vectors)
            total_chunks += len(chunks)
        return len(docs), total_chunks

    async def ingest_directory(self, directory: Path) -> tuple[int, int]:
        docs = [
            Document(doc_id=p.stem, text=p.read_text(encoding="utf-8"), metadata={"source_file": p.name})
            for p in sorted(directory.glob("*"))
            if p.suffix.lower() in {".txt", ".md"} and p.read_text(encoding="utf-8").strip()
        ]
        return await self.ingest(docs) if docs else (0, 0)

    # ---- retrieval ---------------------------------------------------
    async def search(self, query: str, top_k: int | None = None) -> list[Source]:
        k = top_k or self.settings.top_k
        try:
            qvec = await self.embedder.embed([query])
        except Exception as e:  # noqa: BLE001
            raise UpstreamError(f"embedding failed: {e}") from e
        hits = await asyncio.to_thread(self.store.search, qvec, k)
        return [
            Source(doc_id=c.doc_id, chunk_id=c.chunk_id, score=round(s, 4), text=c.text)
            for c, s in hits
            if s >= self.settings.min_score
        ]

    # ---- generation --------------------------------------------------
    async def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        t0 = time.perf_counter()
        sources = await self.search(question, top_k)
        if not sources:
            return QueryResponse(answer=NO_ANSWER, sources=[], latency_ms=_ms(t0))
        try:
            text = await self.llm.generate(question, sources)
        except Exception as e:  # noqa: BLE001
            raise UpstreamError(f"llm failed: {e}") from e
        return QueryResponse(answer=text, sources=sources, latency_ms=_ms(t0))

    async def answer_stream(self, question: str, top_k: int | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yields events: sources -> token* -> done (or error). Sources go first so a UI can show citations early."""
        t0 = time.perf_counter()
        try:
            sources = await self.search(question, top_k)
            yield {"event": "sources", "data": {"sources": [s.model_dump() for s in sources]}}
            if not sources:
                yield {"event": "token", "data": {"text": NO_ANSWER}}
            else:
                async for tok in self.llm.stream(question, sources):
                    yield {"event": "token", "data": {"text": tok}}
            yield {"event": "done", "data": {"latency_ms": _ms(t0)}}
        except Exception as e:  # noqa: BLE001 - headers are already sent; report in-band
            log.exception("stream failed")
            yield {"event": "error", "data": {"message": str(e)}}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
