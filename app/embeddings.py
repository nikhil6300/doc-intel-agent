"""Embedding providers behind one tiny interface so you can swap them without touching the API."""

import re
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.config import Settings

_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been it this that as at by from "
    "what which who how do does did can should must will not no if then than so such".split()
)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder(Protocol):
    signature: str  # changes whenever vectors become incompatible -> triggers re-indexing

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors (cosine == dot product)."""
        ...


class HashEmbedder:
    """Feature-hashing bag of unigrams+bigrams. Offline and deterministic: great for tests and
    CI, but it is lexical, not semantic. Use a real model for anything user-facing."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.signature = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = zlib.crc32(f.encode())  # crc32, not hash(): stable across processes
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return np.sign(v) * np.log1p(np.abs(v))

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))


class OpenAIEmbedder:
    """Works with OpenAI or any OpenAI-compatible server (LiteLLM, vLLM, Azure via base_url)."""

    BATCH = 64

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI  # imported lazily so offline mode needs no key/network

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_s,
        )
        self._model = settings.embedding_model
        self.signature = f"openai-{self._model}"

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            resp = await self._client.embeddings.create(model=self._model, input=list(texts[i : i + self.BATCH]))
            out.extend(d.embedding for d in resp.data)
        return _normalize(np.asarray(out, dtype=np.float32))


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return HashEmbedder(settings.hash_embedding_dim)
