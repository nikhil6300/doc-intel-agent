"""Paragraph/sentence-aware chunking with character overlap.

Why not a fixed window? Cutting mid-sentence hurts retrieval and makes citations ugly.
We pack whole paragraphs (then whole sentences) up to `chunk_size`, and carry a small
overlap into the next chunk so facts that straddle a boundary are not lost.
"""

import re

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text.replace("\r\n", "\n")) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend([p] if len(p) <= chunk_size else _split_long(p, chunk_size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= chunk_size:
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            if tail and len(tail) + 2 + len(piece) <= chunk_size:
                current = f"{tail}\n\n{piece}"
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, hard-splitting giant sentences."""
    out: list[str] = []
    current = ""
    for sentence in _SENT_SPLIT.split(paragraph):
        for part in _hard_split(sentence, chunk_size):
            if not current:
                current = part
            elif len(current) + 1 + len(part) <= chunk_size:
                current = f"{current} {part}"
            else:
                out.append(current)
                current = part
    if current:
        out.append(current)
    return out


def _hard_split(sentence: str, chunk_size: int) -> list[str]:
    if len(sentence) <= chunk_size:
        return [sentence]
    parts: list[str] = []
    current = ""
    for w in sentence.split():
        while len(w) > chunk_size:  # a single "word" longer than a chunk (e.g. a URL / base64)
            if current:
                parts.append(current)
                current = ""
            parts.append(w[:chunk_size])
            w = w[chunk_size:]
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= chunk_size:
            current = f"{current} {w}"
        else:
            parts.append(current)
            current = w
    if current:
        parts.append(current)
    return parts


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")  # start at a word boundary
    return tail[space + 1 :].strip() if space != -1 else tail
