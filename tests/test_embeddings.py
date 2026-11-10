import numpy as np

from app.embeddings import HashEmbedder


async def test_vectors_are_normalised_and_deterministic():
    e = HashEmbedder(256)
    a = await e.embed(["vendor risk assessment"])
    b = await e.embed(["vendor risk assessment"])
    assert a.shape == (1, 256)
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-5)
    assert np.array_equal(a, b)


async def test_related_text_scores_higher_than_unrelated():
    e = HashEmbedder(512)
    q, related, unrelated = await e.embed(
        [
            "how often are tier 1 vendors reassessed",
            "Tier 1 vendors must be reassessed every 12 months",
            "pizza dough needs yeast and flour",
        ]
    )
    assert float(q @ related) > float(q @ unrelated)
