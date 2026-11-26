import pytest

from evals.metrics import (
    generation_summary,
    hit_at_k,
    latency_summary,
    recall_at_k,
    reciprocal_rank,
    retrieval_summary,
)


def test_hit_recall_rr_basic():
    ranked = ["b", "a", "c", "a"]
    assert hit_at_k(ranked, ["a"], 1) == 0.0
    assert hit_at_k(ranked, ["a"], 2) == 1.0
    assert recall_at_k(ranked, ["a", "c"], 2) == 0.5
    assert recall_at_k(ranked, ["a", "c"], 3) == 1.0
    assert reciprocal_rank(ranked, ["a"]) == 0.5
    assert reciprocal_rank(ranked, ["zzz"]) == 0.0


def test_recall_with_no_expected_docs_is_zero_not_a_crash():
    assert recall_at_k(["a"], [], 3) == 0.0


def _rec(i, qtype, answerable, retrieved, expected, source="manual", **extra):
    return {
        "id": i,
        "type": qtype,
        "answerable": answerable,
        "source": source,
        "expected_doc_ids": expected,
        "retrieved": [{"doc_id": d} for d in retrieved],
        **extra,
    }


def test_retrieval_summary_ignores_unanswerable_and_breaks_down_by_type():
    recs = [
        _rec("1", "factoid", True, ["a", "b"], ["a"]),
        _rec("2", "factoid", True, ["b", "a"], ["a"]),
        _rec("3", "numeric", True, ["x", "y"], ["a"]),
        _rec("4", "unanswerable", False, ["a"], []),
    ]
    s = retrieval_summary(recs, (1, 3))
    assert s["n"] == 3
    assert s["hit@1"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["hit@3"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
    assert s["by_type"]["factoid"]["n"] == 2 and s["by_type"]["numeric"]["n"] == 1


def test_generation_summary_refusal_rates():
    recs = [
        _rec("1", "factoid", True, [], ["a"], refused=False),
        _rec("2", "factoid", True, [], ["a"], refused=True),
        _rec("3", "unanswerable", False, [], [], refused=True),
        _rec("4", "unanswerable", False, [], [], refused=False),
    ]
    g = generation_summary(recs)
    assert g["refusal_rate_on_unanswerable"] == 0.5
    assert g["over_refusal_rate"] == 0.5


def test_generation_summary_absent_without_generate():
    g = generation_summary([_rec("1", "factoid", True, [], ["a"])])
    assert g["refusal_rate_on_unanswerable"] is None and g["over_refusal_rate"] is None


def test_latency_summary_percentiles():
    recs = [{"search_ms": float(x)} for x in range(1, 101)]
    s = latency_summary(recs)
    assert s["search_ms"]["p50"] == pytest.approx(50.5, abs=0.6)
    assert s["search_ms"]["p95"] == pytest.approx(95.05, abs=0.6)
