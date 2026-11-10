"""FAISS-backed store. IndexFlatIP over L2-normalised vectors == exact cosine search.

Flat (exact) search is the right default up to ~1e5-1e6 chunks. Week 2 swaps this class for
pgvector behind the same four methods (add_document / search / get_document / list_documents).
"""

import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np

from app.schemas import Chunk, Document, DocumentInfo

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []
        self._docs: dict[str, Document] = {}

    # ---- write path --------------------------------------------------
    def add_document(self, doc: Document, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Insert or *replace* a document (re-ingesting the same doc_id never duplicates)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            if doc.doc_id in self._docs:
                self._remove_locked(doc.doc_id)
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self._chunks.extend(chunks)
            self._docs[doc.doc_id] = doc

    def _remove_locked(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        if self._index is not None and self._index.ntotal:
            all_vecs = self._index.reconstruct_n(0, self._index.ntotal)
            new_index = faiss.IndexFlatIP(all_vecs.shape[1])
            if keep:
                new_index.add(np.ascontiguousarray(all_vecs[keep]))
            self._index = new_index
        self._chunks = [self._chunks[i] for i in keep]
        self._docs.pop(doc_id, None)

    # ---- read path ---------------------------------------------------
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            scores, ids = self._index.search(
                np.ascontiguousarray(query_vec, dtype=np.float32), min(k, self._index.ntotal)
            )
            return [(self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            return self._docs.get(doc_id)

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            counts: dict[str, int] = {}
            for c in self._chunks:
                counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
            return [
                DocumentInfo(doc_id=d.doc_id, chars=len(d.text), chunks=counts.get(d.doc_id, 0), metadata=d.metadata)
                for d in self._docs.values()
            ]

    @property
    def num_chunks(self) -> int:
        with self._lock:
            return len(self._chunks)

    # ---- persistence -------------------------------------------------
    def save(self, directory: Path, signature: str) -> None:
        with self._lock:
            if self._index is None:
                return
            directory.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(directory / "index.faiss"))
            payload = {
                "signature": signature,
                "chunks": [c.model_dump() for c in self._chunks],
                "docs": [d.model_dump() for d in self._docs.values()],
            }
            tmp = directory / "store.json.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(directory / "store.json")  # atomic swap: a crash never leaves a half-written file

    def load(self, directory: Path, signature: str) -> bool:
        """Load a persisted index. Returns False (caller should re-ingest) if missing/incompatible."""
        meta, idx = directory / "store.json", directory / "index.faiss"
        if not (meta.exists() and idx.exists()):
            return False
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            if payload["signature"] != signature:
                log.warning("Index embedding signature %s != %s; re-ingesting", payload["signature"], signature)
                return False
            index = faiss.read_index(str(idx))
            chunks = [Chunk(**c) for c in payload["chunks"]]
            if index.ntotal != len(chunks):
                log.warning("Corrupt index (vectors != chunks); re-ingesting")
                return False
        except Exception:  # noqa: BLE001 - any corruption => rebuild rather than crash on boot
            log.exception("Failed to load persisted index; re-ingesting")
            return False
        with self._lock:
            self._index, self._chunks = index, chunks
            self._docs = {d["doc_id"]: Document(**d) for d in payload["docs"]}
        return True
