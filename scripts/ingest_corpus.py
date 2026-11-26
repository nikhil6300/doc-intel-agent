"""Ingest a folder of PDFs / .txt / .md files into the running Document Intelligence API.

Usage (API must be running):
    python scripts/ingest_corpus.py --dir data/corpus --dry-run     # inspect first, no network
    python scripts/ingest_corpus.py --dir data/corpus               # ingest
    python scripts/ingest_corpus.py --dir data/corpus --api-key $API_KEY --api-url http://localhost:8000

Notes
- doc_id is derived from the file path, so re-running replaces a document instead of duplicating it.
- PDFs that yield almost no text are usually scanned images: they are reported and skipped (OCR is out of scope).
- Keep raw corpus files OUT of git (data/corpus/ is git-ignored): some sources have non-commercial licences.
"""

import argparse
import re
import sys
from pathlib import Path

import httpx
from pypdf import PdfReader

MIN_CHARS = 500  # below this a document is probably a scanned PDF or an empty page range
MAX_CHARS = 2_000_000  # API limit per document


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # undo end-of-line hyphenation: "assess-\nment" -> "assessment"
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # PDF hard line wraps -> spaces (keep blank-line paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return clean(extract_pdf(path))
    return clean(path.read_text(encoding="utf-8", errors="replace"))


def doc_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", "-".join(rel.parts))[:100]


def collect(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("data/corpus"))
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--dry-run", action="store_true", help="extract and report only; do not call the API")
    args = ap.parse_args()

    if not args.dir.exists():
        print(f"ERROR: {args.dir} does not exist", file=sys.stderr)
        return 2
    files = collect(args.dir)
    if not files:
        print(f"ERROR: no .pdf/.txt/.md files under {args.dir}", file=sys.stderr)
        return 2

    docs, skipped = [], []
    print(f"{'doc_id':<60} {'chars':>10}")
    for path in files:
        try:
            text = read_file(path)
        except Exception as e:  # noqa: BLE001 - report and continue with the other files
            skipped.append((path.name, f"unreadable: {e}"))
            continue
        did = doc_id_for(path, args.dir)
        if len(text) < MIN_CHARS:
            skipped.append((path.name, f"only {len(text)} chars (scanned PDF?)"))
            continue
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            print(f"WARNING: {path.name} truncated to {MAX_CHARS} chars")
        collection = path.relative_to(args.dir).parts[0] if len(path.relative_to(args.dir).parts) > 1 else "root"
        docs.append({"doc_id": did, "text": text, "metadata": {"source_file": path.name, "collection": collection}})
        print(f"{did:<60} {len(text):>10}")

    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")
    print(f"\n{len(docs)} documents ready, {len(skipped)} skipped, {sum(len(d['text']) for d in docs):,} chars total")
    if args.dry_run or not docs:
        return 0 if docs else 1

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    total_chunks = 0
    with httpx.Client(base_url=args.api_url, headers=headers, timeout=600) as client:
        for i in range(0, len(docs), 5):  # small batches keep request size and embedding calls reasonable
            batch = docs[i : i + 5]
            resp = client.post("/v1/ingest", json={"documents": batch})
            if resp.status_code != 200:
                print(f"ERROR {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
                return 1
            total_chunks += resp.json()["chunks"]
            print(f"ingested {i + len(batch)}/{len(docs)} documents")
    print(f"Done: {len(docs)} documents -> {total_chunks} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
