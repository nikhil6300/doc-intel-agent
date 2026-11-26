"""Human-in-the-loop core: which answers need a person, and where the review decisions are stored.

Escalation rules (deterministic, so they are testable and explainable to an auditor):
  low_confidence    best retrieved chunk scores below REVIEW_SCORE_THRESHOLD
  uncited           the answer has evidence but cites none of it
  invalid_citation  the answer cites a source number that was never provided
A refusal ("I don't know ...") is never escalated: refusing is the safe behaviour.

Storage is SQLite (stdlib, zero setup, file lives next to the index). It is an append-and-update audit trail:
what was asked, what was answered from which sources, who decided what, and when.
"""

import json
import re
import sqlite3
import statistics
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm import NO_ANSWER
from app.schemas import QueryResponse, Source

WITHHELD_MESSAGE = "This answer is awaiting analyst review. Check GET /v1/answers/{query_id} for the outcome."
_CITATION = re.compile(r"\[(\d+)\]")


class ReviewError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code, self.message = status_code, message


def assess(answer: str, sources: list[Source], score_threshold: float = 0.0) -> tuple[bool, str | None]:
    """Return (needs_review, reason)."""
    if answer.strip().startswith(NO_ANSWER) or not sources:
        return False, None
    top = max(s.score for s in sources)
    if score_threshold > 0 and top < score_threshold:
        return True, "low_confidence"
    cited = [int(n) for n in _CITATION.findall(answer)]
    if not cited:
        return True, "uncited"
    if any(n < 1 or n > len(sources) for n in cited):
        return True, "invalid_citation"
    return False, None


def apply_hitl(settings: Settings, store: "ReviewStore", question: str, resp: QueryResponse) -> QueryResponse:
    """Record the interaction and, in gate mode, withhold answers that need a person."""
    if settings.hitl_mode == "off":
        return resp
    needs, reason = assess(resp.answer, resp.sources, settings.review_score_threshold)
    top = max((s.score for s in resp.sources), default=0.0)
    qid = store.record(
        question=question,
        answer=resp.answer,
        sources=[s.model_dump() for s in resp.sources],
        top_score=top,
        needs_review=needs,
        reason=reason,
    )
    update: dict[str, Any] = {"query_id": qid, "needs_review": needs, "review_reason": reason}
    if needs and settings.hitl_mode == "gate":
        update.update(review_status="withheld", answer=WITHHELD_MESSAGE, sources=[])
    elif needs:
        update["review_status"] = "flagged"
    return resp.model_copy(update=update)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
                    sources TEXT NOT NULL, top_score REAL NOT NULL, needs_review INTEGER NOT NULL,
                    review_reason TEXT, status TEXT NOT NULL, reviewer TEXT, corrected_answer TEXT,
                    review_comment TEXT, reviewed_at TEXT, rating INTEGER, feedback_comment TEXT)"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON interactions(status, created_at)")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- writes ----------------------------------------------------------
    def record(
        self,
        *,
        question: str,
        answer: str,
        sources: list[dict],
        top_score: float,
        needs_review: bool,
        reason: str | None,
    ) -> str:
        qid = uuid.uuid4().hex[:16]
        with self._lock:
            self._conn.execute(
                "INSERT INTO interactions (id, created_at, question, answer, sources, top_score, needs_review,"
                " review_reason, status) VALUES (?,?,?,?,?,?,?,?,?)",
                (qid, _now(), question, answer, json.dumps(sources), top_score, int(needs_review), reason,
                 "pending" if needs_review else "auto"),
            )  # fmt: skip
            self._conn.commit()
        return qid

    def resolve(self, qid: str, decision: str, reviewer: str, corrected: str | None, comment: str | None) -> dict:
        if decision == "edit" and not (corrected and corrected.strip()):
            raise ReviewError(422, "decision 'edit' requires corrected_answer")
        status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[decision]
        with self._lock:
            row = self._conn.execute("SELECT status FROM interactions WHERE id=?", (qid,)).fetchone()
            if row is None:
                raise ReviewError(404, f"query '{qid}' not found")
            if row["status"] != "pending":
                raise ReviewError(409, f"query '{qid}' is not pending review (status: {row['status']})")
            self._conn.execute(
                "UPDATE interactions SET status=?, reviewer=?, corrected_answer=?, review_comment=?, reviewed_at=?"
                " WHERE id=?",
                (status, reviewer, corrected if decision == "edit" else None, comment, _now(), qid),
            )  # fmt: skip
            self._conn.commit()
        return self.get(qid)  # type: ignore[return-value]

    def add_feedback(self, qid: str, rating: int, comment: str | None) -> None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE interactions SET rating=?, feedback_comment=? WHERE id=?", (rating, comment, qid)
            )
            self._conn.commit()
            if cur.rowcount == 0:
                raise ReviewError(404, f"query '{qid}' not found")

    # ---- reads -----------------------------------------------------------
    def get(self, qid: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM interactions WHERE id=?", (qid,)).fetchone()
        return _row(row) if row else None

    def queue(self, status: str = "pending", limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM interactions WHERE status=? ORDER BY created_at ASC LIMIT ?", (status, limit)
            ).fetchall()
        return [_row(r) for r in rows]

    def export(self, statuses: tuple[str, ...] = ("edited", "rejected")) -> list[dict]:
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM interactions WHERE status IN ({marks}) ORDER BY created_at ASC", statuses
            ).fetchall()  # noqa: S608 - placeholders only, values are bound
        return [_row(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM interactions").fetchall()
        items = [_row(r) for r in rows]
        by_status: dict[str, int] = {}
        for i in items:
            by_status[i["status"]] = by_status.get(i["status"], 0) + 1
        decided = [i for i in items if i["status"] in {"approved", "edited", "rejected"}]
        rated = [i["rating"] for i in items if i["rating"] is not None]
        secs = [
            (datetime.fromisoformat(i["reviewed_at"]) - datetime.fromisoformat(i["created_at"])).total_seconds()
            for i in decided
        ]  # fmt: skip

        def rate(n: int, d: int) -> float | None:
            return round(n / d, 4) if d else None

        return {
            "total_queries": len(items),
            "by_status": by_status,
            "escalation_rate": rate(sum(1 for i in items if i["needs_review"]), len(items)),
            "approval_rate": rate(by_status.get("approved", 0), len(decided)),
            "edit_rate": rate(by_status.get("edited", 0), len(decided)),
            "rejection_rate": rate(by_status.get("rejected", 0), len(decided)),
            "median_review_seconds": round(statistics.median(secs), 1) if secs else None,
            "thumbs_up_rate": rate(sum(1 for r in rated if r > 0), len(rated)),
        }


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["query_id"] = d.pop("id")
    d["sources"] = json.loads(d["sources"])
    d["needs_review"] = bool(d["needs_review"])
    return d
