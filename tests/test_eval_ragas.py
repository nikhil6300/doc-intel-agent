import pytest

from evals.ragas_runner import build_rows, estimate_calls

RUN = {
    "items": [
        {
            "id": "1",
            "question": "q1?",
            "answerable": True,
            "refused": False,
            "answer": "a1",
            "contexts": ["c1", "c2"],
            "reference_answer": "r1",
        },
        {
            "id": "2",
            "question": "q2?",
            "answerable": True,
            "refused": True,
            "answer": "I don't know",
            "contexts": [],
            "reference_answer": "r2",
        },
        {
            "id": "3",
            "question": "q3?",
            "answerable": False,
            "refused": True,
            "answer": "I don't know",
            "contexts": [],
            "reference_answer": None,
        },
        {"id": "4", "question": "q4?", "answerable": True, "answer": None, "reference_answer": "r4"},
    ]
}


def test_only_answerable_non_refused_questions_are_scored():
    rows = build_rows(RUN)
    assert [r["id"] for r in rows] == ["1"]  # refused, unanswerable and empty-answer records are all skipped
    assert (
        rows[0]["user_input"] == "q1?"
        and rows[0]["retrieved_contexts"] == ["c1", "c2"]
        and rows[0]["reference"] == "r1"
    )


def test_cost_estimate_formula():
    rows = [{"retrieved_contexts": ["a", "b", "c", "d"]}, {"retrieved_contexts": ["a"]}]
    est = estimate_calls(rows)
    assert est["questions"] == 2
    assert est["judge_calls"] == (2 + 3 + 4 + 1) + (2 + 3 + 1 + 1)
    assert est["embedding_calls"] == 4


def test_ragas_metric_names_resolve():
    pytest.importorskip("ragas")
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    names = [m().name for m in (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall)]
    assert all(isinstance(n, str) and n for n in names)
