import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLE_DIR, docs)
    # Explicit kwargs beat env vars, so tests never touch real providers.
    return Settings(
        embedding_provider="hash",
        llm_provider="extractive",
        openai_api_key=None,
        api_key=None,
        data_dir=docs,
        index_dir=tmp_path / "index",
        auto_ingest=True,
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:  # `with` runs lifespan (startup/shutdown)
        yield c
