"""Quality gate: fail (exit 1) when a run's retrieval metrics fall below minimums.

    python -m evals.gate eval_runs/<run>.json --min hit@3=0.9 mrr=0.8

Used in two places: the free offline test in tests/test_retrieval_gate.py (every pull request) and,
later, the nightly real-model eval in CI.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MINIMUMS = {"hit@3": 0.9, "mrr": 0.8}


def check(summary: dict, minimums: dict[str, float]) -> list[str]:
    retrieval = summary["retrieval"]
    failures = []
    for name, floor in minimums.items():
        if name not in retrieval:
            failures.append(f"{name}: not in the run summary")
        elif retrieval[name] < floor:
            failures.append(f"{name} = {retrieval[name]:.3f} is below the required {floor:.3f}")
    return failures


def parse_minimums(pairs: list[str]) -> dict[str, float]:
    out = {}
    for p in pairs:
        name, _, value = p.partition("=")
        if not name or not value:
            raise ValueError(f"expected name=value, got '{p}'")
        out[name] = float(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_file", type=Path)
    ap.add_argument("--min", nargs="+", default=[f"{k}={v}" for k, v in DEFAULT_MINIMUMS.items()])
    args = ap.parse_args()
    run = json.loads(args.run_file.read_text(encoding="utf-8"))
    failures = check(run["summary"], parse_minimums(args.min))
    for f in failures:
        print(f"GATE FAILED: {f}")
    print("gate passed" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
