"""Stage 5, PR tier: a free, deterministic, offline quality gate.

It runs the demo golden set through the real API (offline hash embedder) and fails the pull request if retrieval
quality drops. It protects chunking, the vector store and search ranking from accidental regressions.
The expensive real-model evaluation runs separately (nightly), never on every PR.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from evals.client import EvalClient
from evals.gate import DEFAULT_MINIMUMS, check
from evals.golden import load_golden
from evals.run_eval import run_evaluation

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "golden_sample.jsonl"


def test_retrieval_quality_gate(client):
    run = run_evaluation(load_golden(SAMPLE), EvalClient(client), label="pr-gate")
    assert check(run["summary"], DEFAULT_MINIMUMS) == []


def test_gate_actually_fails_on_a_degraded_pipeline(settings):
    """Proves the gate can trip: an 8-dimensional embedding collapses retrieval quality."""
    broken = settings.model_copy(update={"hash_embedding_dim": 8})
    with TestClient(create_app(broken)) as c:
        run = run_evaluation(load_golden(SAMPLE), EvalClient(c), label="broken")
    failures = check(run["summary"], DEFAULT_MINIMUMS)
    assert failures, "gate should have failed for a pipeline this degraded"
