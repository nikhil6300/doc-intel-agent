"""Validate the golden set against the documents that are actually ingested.

    python -m evals.validate_golden --golden data/golden.jsonl

Checks: schema, duplicate ids, every expected_doc_id exists, every evidence_quote really appears in an expected
document, and prints the composition (counts per type / source / split). Exit code 1 if anything is wrong.
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from evals.client import EvalClient
from evals.golden import load_golden, split_of
from evals.schemas import GoldenItem


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def validate(items: list[GoldenItem], client: EvalClient) -> list[str]:
    problems: list[str] = []
    known = {d["doc_id"] for d in client.documents()}
    cache: dict[str, str] = {}
    for it in items:
        for d in it.expected_doc_ids:
            if d not in known:
                problems.append(f"{it.id}: expected_doc_id '{d}' is not ingested")
        if it.answerable and not it.evidence_quote:
            problems.append(f"{it.id}: answerable question without evidence_quote")
        if it.evidence_quote and it.expected_doc_ids:
            found = False
            for d in it.expected_doc_ids:
                if d not in known:
                    continue
                if d not in cache:
                    doc = client.document(d)
                    cache[d] = _norm(doc["text"]) if doc else ""
                if _norm(it.evidence_quote) in cache[d]:
                    found = True
                    break
            if not found and any(d in known for d in it.expected_doc_ids):
                problems.append(
                    f"{it.id}: evidence_quote not found in {it.expected_doc_ids}: '{it.evidence_quote[:60]}'"
                )
    return problems


def composition(items: list[GoldenItem]) -> dict[str, Counter]:
    return {
        "type": Counter(i.type for i in items),
        "source": Counter(i.source for i in items),
        "split": Counter(split_of(i.id) for i in items),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden.jsonl"))
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()
    items = load_golden(args.golden)  # raises with file:line on schema problems
    print(f"{len(items)} questions")
    for name, counter in composition(items).items():
        print(f"  by {name}: {dict(sorted(counter.items()))}")
    problems = validate(items, EvalClient.from_url(args.api_url, args.api_key))
    for p in problems:
        print(f"PROBLEM {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
