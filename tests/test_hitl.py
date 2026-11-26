from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.reviews import WITHHELD_MESSAGE, assess
from app.schemas import Source
from evals.client import EvalClient
from evals.promote_feedback import to_candidates

Q = {"question": "How often must Tier 1 vendors be reassessed?"}


def _src(score: float) -> Source:
    return Source(doc_id="d", chunk_id="d::0", score=score, text="t")


def _app(settings: Settings, **update) -> TestClient:
    return TestClient(create_app(settings.model_copy(update=update)))


# ---- escalation rules (pure functions) ------------------------------------
def test_assess_refusal_is_never_escalated():
    assert assess("I don't know based on the provided documents.", [_src(0.1)], 0.9) == (False, None)
    assert assess("anything", [], 0.9) == (False, None)


def test_assess_rules():
    assert assess("Every 12 months [1].", [_src(0.8)], 0.5) == (False, None)
    assert assess("Every 12 months [1].", [_src(0.3)], 0.5) == (True, "low_confidence")
    assert assess("Every 12 months.", [_src(0.8)]) == (True, "uncited")
    assert assess("Every 12 months [3].", [_src(0.8), _src(0.7)]) == (True, "invalid_citation")
    assert assess("Every 12 months [0].", [_src(0.8)]) == (True, "invalid_citation")
    assert assess("Every 12 months [1].", [_src(0.01)], 0.0) == (False, None)  # threshold 0 disables the score rule


# ---- flag mode: serve everything, queue the risky ones ----------------------
def test_flag_mode_serves_and_records_a_good_answer(settings):
    with _app(settings, hitl_mode="flag") as c:
        r = c.post("/v1/query", json=Q).json()
        assert r["query_id"] and r["needs_review"] is False and r["review_status"] == "auto"
        assert "12 months" in r["answer"] and r["sources"]
        status = c.get(f"/v1/answers/{r['query_id']}").json()
        assert status["status"] == "auto" and status["answer"] == r["answer"]
        assert c.get("/v1/review/queue").json() == []


def test_flag_mode_risky_answer_is_served_and_queued(settings):
    with _app(settings, hitl_mode="flag", review_score_threshold=0.99) as c:
        r = c.post("/v1/query", json=Q).json()
        assert r["needs_review"] and r["review_status"] == "flagged" and r["review_reason"] == "low_confidence"
        assert r["sources"] and "12 months" in r["answer"]  # still served
        queue = c.get("/v1/review/queue").json()
        assert [i["query_id"] for i in queue] == [r["query_id"]]
        assert queue[0]["top_score"] > 0 and queue[0]["status"] == "pending"


def test_off_mode_records_nothing(settings):
    with _app(settings, hitl_mode="off") as c:
        r = c.post("/v1/query", json=Q).json()
        assert r["query_id"] is None and r["needs_review"] is False
        assert c.get("/v1/review/stats").json()["total_queries"] == 0


# ---- gate mode: withhold until a person decides -----------------------------
@pytest.fixture
def gate(settings):
    with _app(settings, hitl_mode="gate", review_score_threshold=0.99) as c:
        yield c


def _ask(c):
    r = c.post("/v1/query", json=Q).json()
    assert r["review_status"] == "withheld"
    return r


def test_gate_withholds_answer_and_sources(gate):
    r = _ask(gate)
    assert r["answer"] == WITHHELD_MESSAGE and r["sources"] == [] and r["needs_review"]
    pending = gate.get(f"/v1/answers/{r['query_id']}").json()
    assert pending["status"] == "pending" and pending["answer"] is None


def test_approve_releases_the_original_answer(gate):
    qid = _ask(gate)["query_id"]
    item = gate.get("/v1/review/queue").json()[0]
    assert "12 months" in item["answer"]  # the reviewer sees what was withheld
    d = gate.post(f"/v1/review/{qid}", json={"decision": "approve", "reviewer": "asha"}).json()
    assert d["status"] == "approved" and d["reviewer"] == "asha" and d["reviewed_at"]
    final = gate.get(f"/v1/answers/{qid}").json()
    assert final["status"] == "approved" and "12 months" in final["answer"] and final["sources"]


def test_edit_replaces_the_answer(gate):
    qid = _ask(gate)["query_id"]
    gate.post(f"/v1/review/{qid}", json={"decision": "edit", "reviewer": "asha", "corrected_answer": "Annually [1]."})
    final = gate.get(f"/v1/answers/{qid}").json()
    assert final["status"] == "edited" and final["answer"] == "Annually [1]."


def test_reject_never_releases_an_answer(gate):
    qid = _ask(gate)["query_id"]
    gate.post(f"/v1/review/{qid}", json={"decision": "reject", "reviewer": "asha", "comment": "wrong tier"})
    final = gate.get(f"/v1/answers/{qid}").json()
    assert final["status"] == "rejected" and final["answer"] is None and "rejected" in final["message"]


def test_review_validation_and_state_errors(gate):
    qid = _ask(gate)["query_id"]
    assert gate.post(f"/v1/review/{qid}", json={"decision": "edit", "reviewer": "a"}).status_code == 422
    assert gate.post(f"/v1/review/{qid}", json={"decision": "maybe", "reviewer": "a"}).status_code == 422
    assert gate.post("/v1/review/nope", json={"decision": "approve", "reviewer": "a"}).status_code == 404
    assert gate.post(f"/v1/review/{qid}", json={"decision": "approve", "reviewer": "a"}).status_code == 200
    again = gate.post(f"/v1/review/{qid}", json={"decision": "reject", "reviewer": "b"})
    assert again.status_code == 409  # a decision cannot be silently overwritten
    assert gate.get("/v1/answers/nope").status_code == 404


def test_gate_mode_disables_streaming(gate):
    assert gate.post("/v1/query/stream", json=Q).status_code == 409


# ---- streaming in flag mode ---------------------------------------------------
def test_stream_in_flag_mode_reports_query_id(settings):
    import json

    with _app(settings, hitl_mode="flag", review_score_threshold=0.99) as c:
        with c.stream("POST", "/v1/query/stream", json=Q) as r:
            blocks = r.read().decode().strip().split("\n\n")
        done = json.loads([b for b in blocks if b.startswith("event: done")][0].split("data: ")[1])
        assert done["query_id"] and done["needs_review"] is True and done["review_reason"] == "low_confidence"
        assert c.get("/v1/review/queue").json()[0]["query_id"] == done["query_id"]


# ---- feedback, stats, audit trail -----------------------------------------------
def test_feedback_and_stats(settings):
    with _app(settings, hitl_mode="gate", review_score_threshold=0.99) as c:
        ids = [c.post("/v1/query", json=Q).json()["query_id"] for _ in range(3)]
        c.post(f"/v1/review/{ids[0]}", json={"decision": "approve", "reviewer": "a"})
        c.post(f"/v1/review/{ids[1]}", json={"decision": "edit", "reviewer": "a", "corrected_answer": "x"})
        c.post(f"/v1/review/{ids[2]}", json={"decision": "reject", "reviewer": "a"})
        assert c.post("/v1/feedback", json={"query_id": ids[0], "rating": 1}).status_code == 204
        assert c.post("/v1/feedback", json={"query_id": ids[1], "rating": -1, "comment": "unclear"}).status_code == 204
        s = c.get("/v1/review/stats").json()
        assert s["total_queries"] == 3 and s["escalation_rate"] == 1.0
        assert s["approval_rate"] == s["edit_rate"] == s["rejection_rate"] == pytest.approx(0.3333, abs=1e-3)
        assert s["thumbs_up_rate"] == 0.5 and s["median_review_seconds"] is not None
        assert c.post("/v1/feedback", json={"query_id": "nope", "rating": 1}).status_code == 404
        assert c.post("/v1/feedback", json={"query_id": ids[0], "rating": 5}).status_code == 422


def test_audit_trail_survives_restart(settings):
    with _app(settings, hitl_mode="flag", review_score_threshold=0.99) as c:
        qid = c.post("/v1/query", json=Q).json()["query_id"]
        c.post(f"/v1/review/{qid}", json={"decision": "approve", "reviewer": "asha"})
    with _app(settings, hitl_mode="flag") as c2:
        item = c2.get("/v1/review/queue", params={"status": "approved"}).json()[0]
        assert item["query_id"] == qid and item["reviewer"] == "asha" and item["question"] == Q["question"]


# ---- separate reviewer credential -----------------------------------------------
def test_reviewer_key_is_separate_from_consumer_key(settings):
    with _app(
        settings, api_key="consumer", reviewer_api_key="reviewer", hitl_mode="gate", review_score_threshold=0.99
    ) as c:
        consumer = {"X-API-Key": "consumer"}
        both = {"X-API-Key": "consumer", "X-Reviewer-Key": "reviewer"}
        qid = c.post("/v1/query", json=Q, headers=consumer).json()["query_id"]
        assert c.get(f"/v1/answers/{qid}", headers=consumer).status_code == 200  # consumers may poll
        assert c.get("/v1/review/queue", headers=consumer).status_code == 401  # but not review
        approve = {"decision": "approve", "reviewer": "x"}
        assert c.post(f"/v1/review/{qid}", json=approve, headers=consumer).status_code == 401
        assert c.get("/v1/review/queue", headers={"X-Reviewer-Key": "reviewer"}).status_code == 401  # needs both
        assert c.post(f"/v1/review/{qid}", json=approve, headers=both).status_code == 200


# ---- the flywheel: analyst corrections -> golden-set candidates ----------------------
def test_corrections_become_golden_candidates(settings):
    with _app(settings, hitl_mode="gate", review_score_threshold=0.99) as c:
        a = c.post("/v1/query", json=Q).json()["query_id"]
        b = c.post("/v1/query", json=Q).json()["query_id"]
        c.post(
            f"/v1/review/{a}",
            json={"decision": "edit", "reviewer": "a", "corrected_answer": "Annually.", "comment": "fix"},
        )
        c.post(f"/v1/review/{b}", json={"decision": "reject", "reviewer": "a"})
        rows = to_candidates(EvalClient(c).review_export("edited,rejected"))
    by_dec = {r["analyst_decision"]: r for r in rows}
    assert by_dec["edited"]["reference_answer"] == "Annually." and by_dec["edited"]["analyst_comment"] == "fix"
    assert by_dec["rejected"]["reference_answer"] is None
    assert all(
        r["needs_human_completion"] and r["expected_doc_ids"] == [] and r["source"] == "analyst-feedback" for r in rows
    )
    assert "vendor_risk_policy" in by_dec["edited"]["retrieved_doc_ids_hint"]  # a hint: every doc that was retrieved


def test_export_rejects_bad_status_filter(settings):
    with _app(settings) as c:
        assert c.get("/v1/review/export", params={"statuses": "bogus"}).status_code == 422


def test_review_db_lives_next_to_the_index(settings: Settings):
    with _app(settings):
        pass
    assert Path(settings.index_dir / "review.db").exists()
