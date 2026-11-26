from pathlib import Path

from evals.client import EvalClient
from evals.golden import load_golden
from evals.schemas import GoldenItem
from evals.validate_golden import composition, validate

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "golden_sample.jsonl"


def test_sample_golden_is_valid_against_sample_docs(client):
    assert validate(load_golden(SAMPLE), EvalClient(client)) == []


def test_validator_catches_missing_doc_and_fake_quote(client):
    items = [
        GoldenItem(
            id="a",
            question="What is x here?",
            type="factoid",
            expected_doc_ids=["nope"],
            reference_answer="y",
            evidence_quote="z",
        ),
        GoldenItem(
            id="b",
            question="How often are Tier 1 vendors reassessed?",
            type="factoid",
            expected_doc_ids=["vendor_risk_policy"],
            reference_answer="12 months",
            evidence_quote="a sentence that is not in the document at all",
        ),
        GoldenItem(
            id="c",
            question="What is x here?",
            type="factoid",
            expected_doc_ids=["vendor_risk_policy"],
            reference_answer="y",
        ),
    ]
    problems = validate(items, EvalClient(client))
    text = " | ".join(problems)
    assert "'nope' is not ingested" in text
    assert "b: evidence_quote not found" in text
    assert "c: answerable question without evidence_quote" in text


def test_quote_matching_ignores_case_and_whitespace(client):
    item = GoldenItem(
        id="a",
        question="How often are Tier 1 vendors reassessed?",
        type="factoid",
        expected_doc_ids=["vendor_risk_policy"],
        reference_answer="12 months",
        evidence_quote="TIER 1 vendors   must be reassessed\nevery 12 months",
    )
    assert validate([item], EvalClient(client)) == []


def test_composition_counts():
    c = composition(load_golden(SAMPLE))
    assert c["type"]["unanswerable"] == 3 and sum(c["split"].values()) == 12
