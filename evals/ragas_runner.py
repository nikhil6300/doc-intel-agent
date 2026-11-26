"""Add Ragas generation metrics to a run that was produced with --generate.

    pip install -r requirements-eval.txt
    export OPENAI_API_KEY=...            # judge + embeddings; set OPENAI_BASE_URL for a gateway
    python -m evals.ragas_runner eval_runs/<run>.json --dry-run          # shows how many judge calls it will make
    python -m evals.ragas_runner eval_runs/<run>.json --judge-model <model> --embed-model <model>

Scores are written back into the same JSON under "ragas". Only answerable, non-refused questions are scored.
Cost note: per question roughly 2 (faithfulness) + 3 (answer relevancy) + 1 per context (precision) + 1 (recall)
judge calls, plus 2 embedding calls, so iterate on the dev split with a cheaper judge.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def build_rows(run: dict) -> list[dict]:
    rows = []
    for r in run["items"]:
        if (
            not r["answerable"]
            or r.get("refused")
            or not r.get("answer")
            or not r.get("contexts")
            or not r.get("reference_answer")
        ):
            continue
        rows.append(
            {
                "id": r["id"],
                "user_input": r["question"],
                "retrieved_contexts": r["contexts"],
                "response": r["answer"],
                "reference": r["reference_answer"],
            }
        )
    return rows


def estimate_calls(rows: list[dict]) -> dict:
    judge = sum(2 + 3 + len(r["retrieved_contexts"]) + 1 for r in rows)
    return {"questions": len(rows), "judge_calls": judge, "embedding_calls": 2 * len(rows)}


def run_ragas(rows: list[dict], judge_model: str, embed_model: str, api_key: str, base_url: str | None) -> dict:
    # Imported here so the rest of the repo works without the heavy ragas dependency.
    from openai import AsyncOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    llm = llm_factory(judge_model, client=client, max_tokens=4096)  # 0.4.x needs a client, not just a model name
    embeddings = embedding_factory("openai", model=embed_model, client=client)
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    names = [m.name for m in metrics]

    dataset = EvaluationDataset.from_list([{k: v for k, v in r.items() if k != "id"} for r in rows])
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings, raise_exceptions=False)
    df = result.to_pandas()

    per_item = {}
    for idx, r in enumerate(rows):
        per_item[r["id"]] = {
            n: (None if _isnan(df.iloc[idx][n]) else float(df.iloc[idx][n])) for n in names if n in df.columns
        }
    means = {n: round(float(df[n].mean()), 4) for n in names if n in df.columns}
    failures = {n: int(df[n].isna().sum()) for n in names if n in df.columns}
    return {
        "judge_model": judge_model,
        "embed_model": embed_model,
        "means": means,
        "failures": failures,
        "per_item": per_item,
    }


def _isnan(x: object) -> bool:
    return x != x  # NaN is the only value not equal to itself


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_file", type=Path)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run = json.loads(args.run_file.read_text(encoding="utf-8"))
    if not run["meta"].get("generate"):
        print("This run has no answers. Re-run evals.run_eval with --generate first.", file=sys.stderr)
        return 2
    rows = build_rows(run)
    est = estimate_calls(rows)
    print(
        f"{est['questions']} questions to score -> about {est['judge_calls']} judge calls "
        f"and {est['embedding_calls']} embedding calls"
    )
    if args.dry_run or not rows:
        return 0
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY (and OPENAI_BASE_URL if you use a gateway).", file=sys.stderr)
        return 2
    run["ragas"] = run_ragas(rows, args.judge_model, args.embed_model, api_key, os.environ.get("OPENAI_BASE_URL"))
    args.run_file.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({"means": run["ragas"]["means"], "failures": run["ragas"]["failures"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
