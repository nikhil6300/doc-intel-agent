"""Pydantic models = the contract of the API. FastAPI turns these into validation + OpenAPI docs."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

DocId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.\-]{1,100}$")]


class Document(BaseModel):
    doc_id: DocId
    text: str = Field(min_length=1, max_length=2_000_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    text: str


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1, max_length=100)


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class DocumentInfo(BaseModel):
    doc_id: str
    chars: int
    chunks: int
    metadata: dict[str, str]


class Source(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float


class SearchResponse(BaseModel):
    query: str
    results: list[Source]
