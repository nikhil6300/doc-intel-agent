"""Turn analyst corrections into golden-set CANDIDATES (the human-in-the-loop -> evals flywheel).

    python -m evals.promote_feedback --out data/golden_candidates.jsonl [--statuses edited,rejected]

Each analyst edit or rejection is a real question where the system was wrong. This writes them as candidate rows.
A candidate is NOT yet a golden question: a person must fill expected_doc_ids and evidence_quote (and check the
reference_answer) before moving the row into data/golden.jsonl, where `python -m evals.validate_golden` will check it.
"""

import argparse
import json
import sys
from pathlib import Path

from evals.client import EvalClient


def to_candidates(items: list[dict]) -> list[dict]:
    rows = []
    for it in items:
        edited = it["status"] == "edited"
        rows.append(
            {
                "id": f"fb_{it['query_id']}",
                "question": it["question"],
                "type": "factoid",
                "source": "analyst-feedback",
                "reference_answer": it["corrected_answer"] if edited else None,
                "model_answer": it["answer"],
                "analyst_decision": it["status"],
                "analyst_comment": it["review_comment"],
                "retrieved_doc_ids_hint": sorted({s["doc_id"] for s in it["sources"]}),
                "expected_doc_ids": [],  # TODO(human): fill in
                "evidence_quote": None,  # TODO(human): fill in
                "needs_human_completion": True,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/golden_candidates.jsonl"))
    ap.add_argument("--statuses", default="edited,rejected")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reviewer-key", default=None)
    args = ap.parse_args()
    client = EvalClient.from_url(args.api_url, args.api_key, reviewer_key=args.reviewer_key)
    rows = to_candidates(client.review_export(args.statuses))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} candidate(s) to {args.out}. Complete expected_doc_ids and evidence_quote by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
