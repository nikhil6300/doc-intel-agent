import json
from pathlib import Path

import pytest

from evals.golden import load_golden, select, split_of
from evals.schemas import GoldenItem

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "golden_sample.jsonl"


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_sample_golden_loads_and_has_all_question_types():
    items = load_golden(SAMPLE)
    assert len(items) == 12
    assert {i.type for i in items} == {"factoid", "numeric", "multi_hop", "unanswerable"}
    assert sum(not i.answerable for i in items) == 3


def test_answerable_needs_docs_and_reference():
    with pytest.raises(ValueError, match="expected_doc_ids"):
        GoldenItem(id="a", question="What is x?", type="factoid", reference_answer="y")
    with pytest.raises(ValueError, match="reference_answer"):
        GoldenItem(id="a", question="What is x?", type="factoid", expected_doc_ids=["d"])


def test_unanswerable_must_not_have_docs():
    with pytest.raises(ValueError, match="unanswerable"):
        GoldenItem(id="a", question="What is x?", type="unanswerable", expected_doc_ids=["d"])


def test_errors_name_the_file_and_line(tmp_path):
    good = {"id": "a", "question": "What is x here?", "type": "unanswerable"}
    bad = {"id": "b", "question": "What is y here?", "type": "factoid"}
    with pytest.raises(ValueError, match=r"g\.jsonl:2"):
        load_golden(_write(tmp_path, [good, bad]))


def test_duplicate_ids_rejected(tmp_path):
    row = {"id": "a", "question": "What is x here?", "type": "unanswerable"}
    with pytest.raises(ValueError, match="duplicate id"):
        load_golden(_write(tmp_path, [row, row]))


def test_split_is_deterministic_and_roughly_30_percent_test():
    ids = [f"q{i:04d}" for i in range(2000)]
    first = [split_of(i) for i in ids]
    assert first == [split_of(i) for i in ids]
    test_share = first.count("test") / len(ids)
    assert 0.25 < test_share < 0.35


def test_select_partitions_the_set():
    items = load_golden(SAMPLE)
    dev, test = select(items, "dev"), select(items, "test")
    assert len(dev) + len(test) == len(items)
    assert {i.id for i in dev}.isdisjoint({i.id for i in test})
    with pytest.raises(ValueError):
        select(items, "nonsense")
