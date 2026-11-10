"""FastAPI application factory.

Run:  uvicorn app.main:create_app --factory --reload
Docs: http://localhost:8000/docs
"""

import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.config import Settings, get_settings
from app.embeddings import build_embedder
from app.llm import build_llm
from app.rag import RAGService, UpstreamError
from app.schemas import (
    Document,
    DocumentInfo,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchResponse,
)
from app.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: build heavy objects ONCE and share them via app.state (never per-request).
        embedder, store = build_embedder(settings), VectorStore()
        service = RAGService(settings, embedder, store, build_llm(settings))
        if store.load(settings.index_dir, embedder.signature):
            log.info("Loaded persisted index: %d chunks", store.num_chunks)
        elif settings.auto_ingest and settings.data_dir.exists():
            docs, chunks = await service.ingest_directory(settings.data_dir)
            log.info("Auto-ingested %d docs / %d chunks from %s", docs, chunks, settings.data_dir)
            store.save(settings.index_dir, embedder.signature)
        app.state.settings, app.state.service, app.state.embedder = settings, service, embedder
        yield
        store.save(settings.index_dir, embedder.signature)  # Shutdown: persist

    app = FastAPI(
        title="Document Intelligence API",
        version=__version__,
        description="RAG service with citations, SSE streaming and a retrieval-only endpoint for MCP/agents.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{ms:.1f}"
        log.info("%s %s -> %d %.1fms rid=%s", request.method, request.url.path, response.status_code, ms, request_id)
        return response

    @app.exception_handler(UpstreamError)
    async def upstream_error_handler(_: Request, exc: UpstreamError) -> JSONResponse:
        log.error("upstream error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": "Upstream model provider failed. Try again."})

    # ---- dependencies ------------------------------------------------
    def get_service(request: Request) -> RAGService:
        return request.app.state.service

    async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        expected = request.app.state.settings.api_key
        if expected and not secrets.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ---- ops endpoints (unauthenticated: used by load balancers / k8s probes) ----
    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    async def ready(service: RAGService = Depends(get_service)) -> JSONResponse:
        """Readiness: the index is populated, so we can serve traffic."""
        n = service.store.num_chunks
        return JSONResponse(status_code=200 if n else 503, content={"ready": bool(n), "chunks": n})

    # ---- v1 API --------------------------------------------------------
    v1 = APIRouter(prefix="/v1", tags=["rag"], dependencies=[Depends(require_api_key)])

    @v1.post("/ingest", response_model=IngestResponse)
    async def ingest(body: IngestRequest, service: RAGService = Depends(get_service)) -> IngestResponse:
        docs, chunks = await service.ingest(body.documents)
        service.store.save(service.settings.index_dir, service.embedder.signature)
        return IngestResponse(documents=docs, chunks=chunks)

    @v1.get("/documents", response_model=list[DocumentInfo])
    async def list_documents(service: RAGService = Depends(get_service)) -> list[DocumentInfo]:
        return service.store.list_documents()

    @v1.get("/documents/{doc_id}", response_model=Document)
    async def get_document(doc_id: str, service: RAGService = Depends(get_service)) -> Document:
        doc = service.store.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        return doc

    @v1.get("/search", response_model=SearchResponse)
    async def search(q: str, k: int = 4, service: RAGService = Depends(get_service)) -> SearchResponse:
        if not 3 <= len(q) <= 2000 or not 1 <= k <= 20:
            raise HTTPException(status_code=422, detail="q must be 3-2000 chars and k within 1-20")
        return SearchResponse(query=q, results=await service.search(q, k))

    @v1.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, service: RAGService = Depends(get_service)) -> QueryResponse:
        return await service.answer(body.question, body.top_k)

    @v1.post("/query/stream")
    async def query_stream(body: QueryRequest, service: RAGService = Depends(get_service)) -> StreamingResponse:
        async def sse() -> AsyncIterator[str]:
            async for ev in service.answer_stream(body.question, body.top_k):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"

        return StreamingResponse(
            sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    app.include_router(v1)
    return app
