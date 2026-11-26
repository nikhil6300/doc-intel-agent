"""HTTP endpoints for human-in-the-loop: consumer feedback, answer status, and the analyst review queue."""

import secrets
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from app.reviews import ReviewError, ReviewStore
from app.schemas import AnswerStatus, FeedbackRequest, ReviewDecision, ReviewItem


def get_reviews(request: Request) -> ReviewStore:
    return request.app.state.reviews


async def require_reviewer(request: Request, x_reviewer_key: str | None = Header(default=None)) -> None:
    """Reviewer endpoints need an extra key when REVIEWER_API_KEY is set, so a consumer key cannot approve answers."""
    expected = request.app.state.settings.reviewer_api_key
    if expected and not secrets.compare_digest(x_reviewer_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing reviewer key")


def build_review_router(require_api_key: Callable[..., Any]) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["human-in-the-loop"], dependencies=[Depends(require_api_key)])

    def _call(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except ReviewError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message) from e

    # ---- consumer endpoints ---------------------------------------------
    @router.get("/answers/{query_id}", response_model=AnswerStatus)
    def answer_status(query_id: str, store: ReviewStore = Depends(get_reviews)) -> AnswerStatus:
        """Poll this after a withheld answer. Returns the analyst's final answer once decided."""
        item = store.get(query_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"query '{query_id}' not found")
        st = item["status"]
        if st == "pending":
            return AnswerStatus(query_id=query_id, status=st, answer=None, message="Awaiting analyst review.")
        if st == "rejected":
            return AnswerStatus(
                query_id=query_id, status=st, answer=None, message="An analyst rejected this answer.",
                reviewer=item["reviewer"], reviewed_at=item["reviewed_at"],
            )  # fmt: skip
        final = item["corrected_answer"] if st == "edited" else item["answer"]
        return AnswerStatus(
            query_id=query_id, status=st, answer=final, sources=item["sources"],
            reviewer=item["reviewer"], reviewed_at=item["reviewed_at"],
        )  # fmt: skip

    @router.post("/feedback", status_code=204)
    def feedback(body: FeedbackRequest, store: ReviewStore = Depends(get_reviews)) -> None:
        _call(lambda: store.add_feedback(body.query_id, body.rating, body.comment))

    # ---- reviewer endpoints ------------------------------------------------
    reviewer = [Depends(require_reviewer)]

    @router.get("/review/queue", response_model=list[ReviewItem], dependencies=reviewer)
    def queue(
        status: str = Query("pending", pattern="^(pending|approved|edited|rejected|auto)$"),
        limit: int = Query(50, ge=1, le=200),
        store: ReviewStore = Depends(get_reviews),
    ) -> list[dict]:
        return store.queue(status, limit)

    @router.post("/review/{query_id}", response_model=ReviewItem, dependencies=reviewer)
    def decide(query_id: str, body: ReviewDecision, store: ReviewStore = Depends(get_reviews)) -> dict:
        return _call(lambda: store.resolve(query_id, body.decision, body.reviewer, body.corrected_answer, body.comment))

    @router.get("/review/stats", dependencies=reviewer)
    def stats(store: ReviewStore = Depends(get_reviews)) -> dict:
        return store.stats()

    @router.get("/review/export", response_model=list[ReviewItem], dependencies=reviewer)
    def export(
        statuses: str = Query("edited,rejected", description="comma-separated: approved, edited, rejected"),
        store: ReviewStore = Depends(get_reviews),
    ) -> list[dict]:
        wanted = tuple(s for s in statuses.split(",") if s in {"approved", "edited", "rejected"})
        if not wanted:
            raise HTTPException(status_code=422, detail="statuses must include approved, edited or rejected")
        return store.export(wanted)

    return router
