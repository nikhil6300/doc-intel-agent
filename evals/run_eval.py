"""Run the golden set against a live API and save a results file.

    python -m evals.run_eval --golden data/golden.jsonl --split dev --label baseline
    python -m evals.run_eval --golden data/golden.jsonl --split dev --label hybrid --generate
    python -m evals.run_eval --golden data/golden_sample.jsonl --label demo      # offline demo set

Retrieval metrics need no LLM. --generate also calls /v1/query for every question (costs model calls) to measure
refusal behaviour and to collect answers+contexts for the Ragas step (python -m evals.ragas_runner).
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.client import EvalClient
from evals.golden import load_golden, select, split_of
from evals.metrics import generation_summary, latency_summary, retrieval_summary
from evals.schemas import GoldenItem

REFUSAL_PREFIX = "I don't know"  # must match NO_ANSWER in app/llm.py
DEFAULT_KS = (1, 3, 5, 10)


def run_evaluation(
    items: list[GoldenItem],
    client: EvalClient,
    ks: tuple[int, ...] = DEFAULT_KS,
    generate: bool = False,
    top_k: int | None = None,
    label: str = "run",
    split: str = "all",
    note: str = "",
) -> dict:
    max_k = max(ks)
    records: list[dict] = []
    for item in items:
        hits, search_ms = client.search(item.question, max_k)
        rec: dict = {
            "id": item.id,
            "question": item.question,
            "type": item.type,
            "source": item.source,
            "split": split_of(item.id),
            "answerable": item.answerable,
            "expected_doc_ids": item.expected_doc_ids,
            "reference_answer": item.reference_answer,
            "retrieved": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "score": h["score"]} for h in hits],
            "search_ms": search_ms,
        }
        if generate:
            resp, query_ms = client.query(item.question, top_k)
            rec["answer"] = resp["answer"]
            rec["contexts"] = [s["text"] for s in resp["sources"]]
            rec["refused"] = resp["answer"].strip().startswith(REFUSAL_PREFIX)
            rec["query_ms"] = query_ms
        records.append(rec)

    return {
        "meta": {
            "label": label,
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "split": split,
            "n_items": len(records),
            "ks": list(ks),
            "generate": generate,
            "note": note,
            "api_version": client.health().get("version"),
            "git_sha": _git_sha(),
        },
        "summary": {
            "retrieval": retrieval_summary(records, ks),
            "generation": generation_summary(records) if generate else None,
            "latency": latency_summary(records),
        },
        "items": records,
    }


def format_markdown(run: dict) -> str:
    m, s = run["meta"], run["summary"]
    r, ks = s["retrieval"], m["ks"]
    lines = [
        f"### {m['label']} ({m['split']} split, {m['n_items']} questions, {m['timestamp_utc']})",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k in ks:
        lines.append(f"| hit@{k} | {r[f'hit@{k}']:.3f} |")
    for k in ks:
        lines.append(f"| recall@{k} | {r[f'recall@{k}']:.3f} |")
    lines.append(f"| MRR | {r['mrr']:.3f} |")
    g = s.get("generation")
    if g:
        lines.append(
            f"| refusal rate on unanswerable (n={g['n_unanswerable']}) | {_fmt(g['refusal_rate_on_unanswerable'])} |"
        )
        lines.append(f"| over-refusal rate on answerable (n={g['n_answerable']}) | {_fmt(g['over_refusal_rate'])} |")
    for key, v in s.get("latency", {}).items():
        lines.append(f"| {key} p50 / p95 | {v['p50']} / {v['p95']} ms |")
    if r["by_type"]:
        lines += ["", "| Question type | n | hit@5 | MRR |", "|---|---|---|---|"]
        for t, a in r["by_type"].items():
            lines.append(f"| {t} | {a['n']} | {a.get('hit@5', 0):.3f} | {a['mrr']:.3f} |")
    return "\n".join(lines)


def save_run(run: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = run["meta"]
    path = out_dir / f"{m['timestamp_utc']}_{m['label']}.json"
    path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(format_markdown(run) + "\n", encoding="utf-8")
    return path


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - not a git checkout / git missing
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden.jsonl"))
    ap.add_argument("--split", choices=["all", "dev", "test"], default="dev")
    ap.add_argument("--label", default="run")
    ap.add_argument("--note", default="", help="free text: what changed in this run (config, model, chunk size...)")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_runs"))
    args = ap.parse_args()

    items = select(load_golden(args.golden), args.split)
    if not items:
        print(f"No questions in split '{args.split}'", file=sys.stderr)
        return 2
    if args.split == "test":
        print("NOTE: you are using the held-out test split. Use it once per milestone, never for tuning.")
    client = EvalClient.from_url(args.api_url, args.api_key)
    run = run_evaluation(
        items, client, generate=args.generate, top_k=args.top_k, label=args.label, split=args.split, note=args.note
    )
    path = save_run(run, args.out_dir)
    print(format_markdown(run))
    print(f"\nSaved {path} (and .md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
