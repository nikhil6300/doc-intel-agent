import pytest

from app.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", 100, 10) == ["Hello world."]


def test_chunks_never_exceed_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_carries_context_forward():
    text = "\n\n".join(f"Sentence number {i} has some unique marker{i}." for i in range(30))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    # the start of chunk n+1 repeats the tail of chunk n
    last_words = chunks[0].split()[-2:]
    assert all(w in chunks[1] for w in last_words)


def test_giant_token_is_hard_split():
    chunks = chunk_text("x" * 1000, chunk_size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks) == "x" * 1000


def test_no_content_lost_without_overlap():
    text = "\n\n".join(f"Fact {i}: value {i * 7}." for i in range(40))
    chunks = chunk_text(text, chunk_size=150, overlap=0)
    joined = " ".join(chunks)
    assert all(f"Fact {i}:" in joined for i in range(40))


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_args(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", size, overlap)
