"""Compare two runs question-by-question, with a paired bootstrap confidence interval.

    python -m evals.compare eval_runs/<baseline>.json eval_runs/<candidate>.json

A difference only counts if the 95% interval excludes zero. With ~60-90 questions most small gains do NOT.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evals.metrics import hit_at_k, recall_at_k, reciprocal_rank


def per_item(run: dict, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in run["items"]:
        if not r["answerable"]:
            continue
        ids = [h["doc_id"] for h in r["retrieved"]]
        if metric == "mrr":
            out[r["id"]] = reciprocal_rank(ids, r["expected_doc_ids"])
        elif metric.startswith("hit@"):
            out[r["id"]] = hit_at_k(ids, r["expected_doc_ids"], int(metric[4:]))
        elif metric.startswith("recall@"):
            out[r["id"]] = recall_at_k(ids, r["expected_doc_ids"], int(metric[7:]))
        else:
            raise ValueError(f"unknown metric {metric}")
    return out


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("the two runs share no question ids")
    diff = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": len(ids),
        "baseline": float(np.mean([a[i] for i in ids])),
        "candidate": float(np.mean([b[i] for i in ids])),
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--metrics", nargs="+", default=["hit@1", "hit@5", "recall@5", "mrr"])
    args = ap.parse_args()
    a_run, b_run = (json.loads(p.read_text(encoding="utf-8")) for p in (args.baseline, args.candidate))
    print(f"baseline : {a_run['meta']['label']} ({a_run['meta']['timestamp_utc']})")
    print(f"candidate: {b_run['meta']['label']} ({b_run['meta']['timestamp_utc']})\n")
    print(f"{'metric':<10} {'n':>4} {'baseline':>9} {'candidate':>10} {'delta':>8}  {'95% CI':<18} verdict")
    for metric in args.metrics:
        r = paired_bootstrap(per_item(a_run, metric), per_item(b_run, metric))
        verdict = "real change" if r["significant"] else "within noise"
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        row = f"{metric:<10} {r['n']:>4} {r['baseline']:>9.3f} {r['candidate']:>10.3f} {r['delta']:>+8.3f}"
        print(f"{row}  {ci:<18} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
