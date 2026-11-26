import numpy as np
import pytest

from evals.compare import paired_bootstrap, per_item


def test_clear_improvement_is_significant():
    a = {f"q{i}": 0.0 for i in range(40)}
    b = {f"q{i}": 1.0 for i in range(40)}
    r = paired_bootstrap(a, b)
    assert r["delta"] == 1.0 and r["significant"] and r["ci_low"] > 0


def test_tiny_noisy_change_is_within_noise():
    rng = np.random.default_rng(1)
    a = {f"q{i}": float(rng.integers(0, 2)) for i in range(30)}
    b = dict(a)
    b["q0"] = 1.0 - a["q0"]  # a single flipped question
    r = paired_bootstrap(a, b)
    assert not r["significant"]


def test_only_shared_ids_are_compared():
    r = paired_bootstrap({"a": 0.0, "b": 0.0, "only_a": 1.0}, {"a": 1.0, "b": 1.0, "only_b": 0.0})
    assert r["n"] == 2


def test_no_overlap_raises():
    with pytest.raises(ValueError):
        paired_bootstrap({"a": 1.0}, {"b": 1.0})


def test_per_item_metrics_skip_unanswerable():
    run = {
        "items": [
            {"id": "1", "answerable": True, "expected_doc_ids": ["a"], "retrieved": [{"doc_id": "b"}, {"doc_id": "a"}]},
            {"id": "2", "answerable": False, "expected_doc_ids": [], "retrieved": [{"doc_id": "a"}]},
        ]
    }
    assert per_item(run, "mrr") == {"1": 0.5}
    assert per_item(run, "hit@1") == {"1": 0.0}
    assert per_item(run, "recall@2") == {"1": 1.0}
