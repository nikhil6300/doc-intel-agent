import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["chunks"] >= 3  # sample docs auto-ingested


def test_list_and_get_document(client):
    docs = client.get("/v1/documents").json()
    ids = {d["doc_id"] for d in docs}
    assert {"vendor_risk_policy", "call_quality_guidelines", "incident_response_runbook"} <= ids
    doc = client.get("/v1/documents/vendor_risk_policy").json()
    assert "Tier 1" in doc["text"]


def test_get_missing_document_is_404(client):
    assert client.get("/v1/documents/nope").status_code == 404


def test_search_ranks_the_right_document_first(client):
    r = client.get("/v1/search", params={"q": "how often must tier 1 vendors be reassessed", "k": 3}).json()
    assert r["results"][0]["doc_id"] == "vendor_risk_policy"
    scores = [x["score"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)


def test_query_returns_cited_answer_and_sources(client):
    r = client.post("/v1/query", json={"question": "Within how many days must a post-incident review be published?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["doc_id"] == "incident_response_runbook"
    assert "five working days" in body["answer"] and "[" in body["answer"]
    assert body["latency_ms"] > 0
    assert "x-request-id" in r.headers


def test_validation_errors_are_422(client):
    assert client.post("/v1/query", json={"question": "hi"}).status_code == 422
    assert client.post("/v1/query", json={"question": "valid question?", "top_k": 99}).status_code == 422
    assert client.post("/v1/ingest", json={"documents": [{"doc_id": "bad id/../x", "text": "t"}]}).status_code == 422
    assert client.get("/v1/search", params={"q": "ab"}).status_code == 422


def test_ingest_then_query_new_document(client):
    payload = {
        "documents": [{"doc_id": "parking", "text": "Employees may park in garage level B2 after 6pm on weekdays."}]
    }
    assert client.post("/v1/ingest", json=payload).json() == {"documents": 1, "chunks": 1}
    r = client.post("/v1/query", json={"question": "Where can employees park after 6pm?"}).json()
    assert r["sources"][0]["doc_id"] == "parking"


def test_reingest_replaces_instead_of_duplicating(client):
    before = {d["doc_id"]: d for d in client.get("/v1/documents").json()}
    client.post("/v1/ingest", json={"documents": [{"doc_id": "vendor_risk_policy", "text": "Short replacement text."}]})
    after = {d["doc_id"]: d for d in client.get("/v1/documents").json()}
    assert after["vendor_risk_policy"]["chunks"] == 1
    assert len(after) == len(before)
    # search must not return stale chunks of the old version
    hits = client.get("/v1/search", params={"q": "SOC 2 Type II report", "k": 10}).json()["results"]
    assert all("SOC 2" not in h["text"] for h in hits)


def test_stream_emits_sources_tokens_done(client):
    with client.stream("POST", "/v1/query/stream", json={"question": "What are the severity 1 update intervals?"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = []
        for block in r.read().decode().strip().split("\n\n"):
            name, data = block.split("\n")
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    names = [n for n, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and "token" in names
    assert events[0][1]["sources"][0]["doc_id"] == "incident_response_runbook"


def test_min_score_filters_irrelevant_queries(settings: Settings):
    strict = settings.model_copy(update={"min_score": 0.5})
    with TestClient(create_app(strict)) as c:
        r = c.post("/v1/query", json={"question": "recipe for chocolate lava cake"}).json()
    assert r["sources"] == [] and r["answer"].startswith("I don't know")


def test_api_key_enforced_on_v1_but_not_health(settings: Settings):
    with TestClient(create_app(settings.model_copy(update={"api_key": "s3cret"}))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/documents").status_code == 401
        assert c.get("/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/documents", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_index_persists_across_restarts(settings: Settings, tmp_path: Path):
    with TestClient(create_app(settings)) as c:
        n = c.get("/ready").json()["chunks"]
    # restart with auto_ingest OFF: data must come from the persisted index
    with TestClient(create_app(settings.model_copy(update={"auto_ingest": False}))) as c2:
        assert c2.get("/ready").json()["chunks"] == n
        assert c2.get("/v1/search", params={"q": "tier 1 vendors"}).json()["results"]


def test_incompatible_index_is_rebuilt(settings: Settings):
    with TestClient(create_app(settings)):
        pass
    changed = settings.model_copy(update={"hash_embedding_dim": 128})  # different signature
    with TestClient(create_app(changed)) as c:
        assert c.get("/ready").json()["chunks"] >= 3  # re-ingested, not crashed
