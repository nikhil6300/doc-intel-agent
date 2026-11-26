"""Pydantic models = the contract of the API. FastAPI turns these into validation + OpenAPI docs."""

from typing import Annotated, Literal

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
    # Human-in-the-loop fields (query_id is None when HITL_MODE=off)
    query_id: str | None = None
    needs_review: bool = False
    review_status: Literal["auto", "flagged", "withheld"] = "auto"
    review_reason: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[Source]


# ---- human-in-the-loop ------------------------------------------------------
class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=64)
    rating: Literal[-1, 1]  # thumbs down / thumbs up
    comment: str | None = Field(default=None, max_length=2000)


class ReviewDecision(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    corrected_answer: str | None = Field(default=None, max_length=10000)  # required for 'edit'
    comment: str | None = Field(default=None, max_length=2000)


class ReviewItem(BaseModel):
    query_id: str
    created_at: str
    question: str
    answer: str
    sources: list[Source]
    top_score: float
    review_reason: str | None
    status: Literal["auto", "pending", "approved", "edited", "rejected"]
    reviewer: str | None = None
    corrected_answer: str | None = None
    review_comment: str | None = None
    reviewed_at: str | None = None
    rating: int | None = None
    feedback_comment: str | None = None


class AnswerStatus(BaseModel):
    query_id: str
    status: Literal["auto", "pending", "approved", "edited", "rejected"]
    answer: str | None  # None while pending or when rejected
    message: str | None = None
    sources: list[Source] = Field(default_factory=list)
    reviewer: str | None = None
    reviewed_at: str | None = None
