import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_corpus  # noqa: E402
import make_manifest as mm  # noqa: E402


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "rbi").mkdir(parents=True)
    (root / "financebench").mkdir()
    (root / "rbi" / "Outsourcing of IT Services (2023).pdf").write_bytes(b"%PDF-1.4 fake rbi")
    (root / "financebench" / "BOEING_2022_10K.pdf").write_bytes(b"%PDF-1.4 fake boeing")
    (root / "notes.md").write_text("# notes\n" + "text " * 200)
    (root / "ignored.docx").write_bytes(b"not scanned")
    return root


def test_doc_id_matches_the_ingestion_script_exactly(corpus):
    for p in (x for x in corpus.rglob("*") if x.suffix in mm.EXTENSIONS):
        assert mm.doc_id_for(p, corpus) == ingest_corpus.doc_id_for(p, corpus)


def test_scan_computes_fields_and_skips_unsupported_files(corpus):
    rows = {r["file"]: r for r in mm.scan(corpus)}
    assert set(rows) == {"rbi/Outsourcing of IT Services (2023).pdf", "financebench/BOEING_2022_10K.pdf", "notes.md"}
    b = rows["financebench/BOEING_2022_10K.pdf"]
    assert b["doc_id"] == "financebench-BOEING_2022_10K" and b["collection"] == "financebench"
    assert len(b["sha256"]) == 64 and "CC BY-NC" in b["licence"] and "financebench" in b["source_url"]
    assert rows["notes.md"]["collection"] == "root"
    assert rows["rbi/Outsourcing of IT Services (2023).pdf"]["doc_id"].startswith("rbi-Outsourcing_of_IT_Services")


def test_rerun_preserves_human_edits_but_refreshes_computed_fields(corpus, tmp_path):
    out = tmp_path / "m.csv"
    rows, _ = mm.merge(mm.scan(corpus), {})
    mm.write_manifest(out, rows)
    # a human fills in the url and a note, then the file content changes
    edited = mm.read_manifest(out)
    key = "rbi/Outsourcing of IT Services (2023).pdf"
    edited[key]["source_url"] = "https://www.rbi.org.in/x"
    edited[key]["notes"] = "Apr 2023 version"
    mm.write_manifest(out, list(edited.values()))
    (corpus / key).write_bytes(b"%PDF-1.4 a newer version")
    merged, dropped = mm.merge(mm.scan(corpus), mm.read_manifest(out))
    row = {r["file"]: r for r in merged}[key]
    assert row["source_url"] == "https://www.rbi.org.in/x" and row["notes"] == "Apr 2023 version"
    assert row["sha256"] == mm.sha256_of(corpus / key) and dropped == []


def test_check_flags_blank_url_changed_file_and_missing_file(corpus, tmp_path):
    out = tmp_path / "m.csv"
    rows, _ = mm.merge(mm.scan(corpus), {})
    mm.write_manifest(out, rows)
    problems = " | ".join(mm.check(mm.scan(corpus), mm.read_manifest(out)))
    assert "source_url is blank" in problems  # rbi and notes.md have no default url
    # fill every url, then everything passes
    filled = mm.read_manifest(out)
    for r in filled.values():
        r["source_url"] = r["source_url"] or "https://example.org"
        r["licence"] = r["licence"] or "public"
    mm.write_manifest(out, list(filled.values()))
    assert mm.check(mm.scan(corpus), mm.read_manifest(out)) == []
    # change a file and delete another
    (corpus / "notes.md").write_text("changed")
    (corpus / "financebench" / "BOEING_2022_10K.pdf").unlink()
    problems = " | ".join(mm.check(mm.scan(corpus), mm.read_manifest(out)))
    assert "sha256 differs" in problems and "not found on disk" in problems


def test_new_file_is_reported_until_manifest_is_regenerated(corpus, tmp_path):
    out = tmp_path / "m.csv"
    rows, _ = mm.merge(mm.scan(corpus), {})
    mm.write_manifest(out, rows)
    (corpus / "nist").mkdir()
    (corpus / "nist" / "SP800-61r3.pdf").write_bytes(b"%PDF-1.4 nist")
    assert any("not in the manifest" in p for p in mm.check(mm.scan(corpus), mm.read_manifest(out)))


def test_csv_has_the_documented_columns(corpus, tmp_path):
    out = tmp_path / "m.csv"
    mm.write_manifest(out, mm.scan(corpus))
    with out.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == mm.FIELDS
