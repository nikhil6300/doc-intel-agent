"""Create / refresh data/corpus_manifest.csv: the provenance record for every document you ingest.

    python scripts/make_manifest.py --dir data/corpus --out data/corpus_manifest.csv
    python scripts/make_manifest.py --check            # exit 1 if a source_url is blank or the manifest is out of date

Columns: file, collection, doc_id, size_kb, sha256, date_downloaded, source_url, licence, notes
- file / collection / doc_id / size_kb / sha256 are computed (doc_id matches scripts/ingest_corpus.py exactly,
  so it is the id you put in golden.jsonl expected_doc_ids).
- date_downloaded, source_url, licence, notes are YOURS: re-running never overwrites what you filled in.
- Commit the manifest, never the documents (data/corpus/ is git-ignored; some licences forbid redistribution).
"""

import argparse
import csv
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

FIELDS = ["file", "collection", "doc_id", "size_kb", "sha256", "date_downloaded", "source_url", "licence", "notes"]
COMPUTED = {"file", "collection", "doc_id", "size_kb", "sha256"}
EXTENSIONS = {".pdf", ".txt", ".md"}

# Starting points only. Replace with what the document itself says if it differs.
DEFAULTS = {
    "financebench": {
        "source_url": "https://github.com/patronus-ai/financebench",
        "licence": "CC BY-NC 4.0 (non-commercial; do not redistribute)",
    },
    "nist": {"licence": "NIST publication (US government work); verify the notice in the document"},
    "rbi": {"licence": "RBI public document; do not redistribute"},
}


def doc_id_for(path: Path, root: Path) -> str:
    """MUST stay identical to doc_id_for in scripts/ingest_corpus.py (a test enforces this)."""
    rel = path.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", "-".join(rel.parts))[:100]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(root: Path) -> list[dict]:
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in EXTENSIONS):
        rel = p.relative_to(root)
        collection = rel.parts[0] if len(rel.parts) > 1 else "root"
        rows.append(
            {
                "file": rel.as_posix(),
                "collection": collection,
                "doc_id": doc_id_for(p, root),
                "size_kb": str(round(p.stat().st_size / 1024)),
                "sha256": sha256_of(p),
                "date_downloaded": date.fromtimestamp(p.stat().st_mtime).isoformat(),
                "source_url": DEFAULTS.get(collection, {}).get("source_url", ""),
                "licence": DEFAULTS.get(collection, {}).get("licence", ""),
                "notes": "",
            }
        )
    return rows


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def merge(scanned: list[dict], existing: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Computed fields come from the files; the human-owned fields come from the existing manifest."""
    merged = []
    for row in scanned:
        old = existing.get(row["file"])
        if old:
            for k in FIELDS:
                if k not in COMPUTED and old.get(k):
                    row[k] = old[k]
        merged.append(row)
    dropped = sorted(set(existing) - {r["file"] for r in scanned})
    return merged, dropped


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def check(scanned: list[dict], existing: dict[str, dict]) -> list[str]:
    problems = []
    for row in scanned:
        old = existing.get(row["file"])
        if old is None:
            problems.append(f"{row['file']}: not in the manifest (run make_manifest.py)")
            continue
        if not old.get("source_url", "").strip():
            problems.append(f"{row['file']}: source_url is blank")
        if not old.get("licence", "").strip():
            problems.append(f"{row['file']}: licence is blank")
        if old.get("sha256") != row["sha256"]:
            problems.append(f"{row['file']}: file changed since the manifest was written (sha256 differs)")
    for gone in sorted(set(existing) - {r["file"] for r in scanned}):
        problems.append(f"{gone}: listed in the manifest but not found on disk")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus_manifest.csv"))
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()
    if not args.dir.exists():
        print(f"ERROR: {args.dir} does not exist", file=sys.stderr)
        return 2
    scanned = scan(args.dir)
    if not scanned:
        print(f"ERROR: no .pdf/.txt/.md files under {args.dir}", file=sys.stderr)
        return 2
    existing = read_manifest(args.out)
    if args.check:
        problems = check(scanned, existing)
        for p in problems:
            print(f"PROBLEM {p}")
        print("manifest OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    rows, dropped = merge(scanned, existing)
    write_manifest(args.out, rows)
    blank = sum(1 for r in rows if not r["source_url"].strip())
    print(f"Wrote {len(rows)} rows to {args.out}")
    for d in dropped:
        print(f"WARNING: dropped row for missing file {d}")
    if blank:
        print(f"{blank} row(s) still need a source_url: open the CSV and fill them in, then run with --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
