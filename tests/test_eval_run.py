import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from evals.client import EvalClient
from evals.golden import load_golden
from evals.run_eval import format_markdown, run_evaluation, save_run

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "golden_sample.jsonl"


def test_retrieval_only_run(client):
    run = run_evaluation(load_golden(SAMPLE), EvalClient(client), label="t")
    r = run["summary"]["retrieval"]
    assert r["n"] == 9  # 12 questions, 3 unanswerable are not retrieval-scored
    assert r["hit@3"] >= 0.9
    assert run["summary"]["generation"] is None
    assert "answer" not in run["items"][0]
    assert run["meta"]["generate"] is False and run["meta"]["api_version"]


def test_generate_run_records_answers_contexts_and_refusals(settings):
    strict = settings.model_copy(update={"min_score": 0.15})
    with TestClient(create_app(strict)) as c:
        run = run_evaluation(load_golden(SAMPLE), EvalClient(c), generate=True, label="gen")
    g = run["summary"]["generation"]
    assert g["over_refusal_rate"] == 0.0  # answerable questions are not wrongly refused at this threshold
    assert g["refusal_rate_on_unanswerable"] >= 0.5
    answered = [r for r in run["items"] if r["answerable"]]
    assert all(r["contexts"] and r["answer"] for r in answered)
    assert "query_ms" in run["summary"]["latency"]


def test_min_score_trades_over_refusal_for_refusal(settings):
    """The central tuning trade-off, visible in numbers: a stricter threshold refuses more of everything."""

    def rates(min_score: float):
        with TestClient(create_app(settings.model_copy(update={"min_score": min_score}))) as c:
            g = run_evaluation(load_golden(SAMPLE), EvalClient(c), generate=True)["summary"]["generation"]
        return g["refusal_rate_on_unanswerable"], g["over_refusal_rate"]

    loose, strict = rates(0.0), rates(0.25)
    assert strict[0] >= loose[0]  # refuses more unanswerable questions
    assert strict[1] >= loose[1]  # but also refuses more answerable ones


def test_save_run_writes_json_and_markdown(client, tmp_path):
    run = run_evaluation(load_golden(SAMPLE), EvalClient(client), label="saved", split="all")
    path = save_run(run, tmp_path / "eval_runs")
    assert path.exists() and path.with_suffix(".md").exists()
    assert json.loads(path.read_text())["meta"]["label"] == "saved"
    md = format_markdown(run)
    assert "| hit@5 |" in md and "| MRR |" in md
