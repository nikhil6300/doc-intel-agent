"""Retrieval metrics at chunk level: k = how many chunks the LLM would be shown.

hit@k     1 if any of the top-k chunks comes from an expected document
recall@k  fraction of the expected documents that appear in the top-k chunks
rr        reciprocal rank of the first chunk from an expected document (mean = MRR)
"""

import numpy as np


def hit_at_k(ranked_doc_ids: list[str], expected: list[str], k: int) -> float:
    return 1.0 if any(d in set(expected) for d in ranked_doc_ids[:k]) else 0.0


def recall_at_k(ranked_doc_ids: list[str], expected: list[str], k: int) -> float:
    exp = set(expected)
    return len(exp & set(ranked_doc_ids[:k])) / len(exp) if exp else 0.0


def reciprocal_rank(ranked_doc_ids: list[str], expected: list[str]) -> float:
    exp = set(expected)
    for rank, d in enumerate(ranked_doc_ids, 1):
        if d in exp:
            return 1.0 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def _mean(xs: list[float]) -> float:
    return round(float(np.mean(xs)), 4) if xs else 0.0


def retrieval_summary(records: list[dict], ks: tuple[int, ...]) -> dict:
    """Aggregate over answerable records only (unanswerable ones have nothing to retrieve)."""
    rows = [r for r in records if r["answerable"]]

    def agg(subset: list[dict]) -> dict:
        out: dict = {"n": len(subset)}
        for k in ks:
            out[f"hit@{k}"] = _mean([hit_at_k(_ids(r), r["expected_doc_ids"], k) for r in subset])
            out[f"recall@{k}"] = _mean([recall_at_k(_ids(r), r["expected_doc_ids"], k) for r in subset])
        out["mrr"] = _mean([reciprocal_rank(_ids(r), r["expected_doc_ids"]) for r in subset])
        return out

    summary = agg(rows)
    summary["by_type"] = {t: agg([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
    summary["by_source"] = {s: agg([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})}
    return summary


def generation_summary(records: list[dict]) -> dict:
    """Refusal behaviour. Needs records produced with --generate."""
    gen = [r for r in records if "refused" in r]
    ans = [r for r in gen if r["answerable"]]
    unans = [r for r in gen if not r["answerable"]]
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        # of the questions that SHOULD be refused, how many were refused (higher is better)
        "refusal_rate_on_unanswerable": _mean([1.0 if r["refused"] else 0.0 for r in unans]) if unans else None,
        # of the questions that should be answered, how many were wrongly refused (lower is better)
        "over_refusal_rate": _mean([1.0 if r["refused"] else 0.0 for r in ans]) if ans else None,
    }


def latency_summary(records: list[dict]) -> dict:
    out = {}
    for key in ("search_ms", "query_ms"):
        vals = [r[key] for r in records if key in r]
        if vals:
            out[key] = {"p50": round(percentile(vals, 50), 1), "p95": round(percentile(vals, 95), 1)}
    return out


def _ids(record: dict) -> list[str]:
    return [h["doc_id"] for h in record["retrieved"]]
