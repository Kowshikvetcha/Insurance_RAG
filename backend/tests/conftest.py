from pathlib import Path

import pytest

from app.config import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "sample_policy.pdf"


def offline_settings(**overrides) -> Settings:
    base = dict(llm_provider="fake", embed_provider="fake", rerank_provider="none")
    return Settings(_env_file=None, **(base | overrides))


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    if not FIXTURE.exists():
        from tests.fixtures.make_sample_policy import make

        make()
    return FIXTURE.read_bytes()


@pytest.fixture(scope="session")
def sample_session(sample_pdf):
    """The full ingestion pipeline (real Docling models, fake LLM/embeddings), run once."""
    from app.ingestion.pipeline import ingest

    return ingest(sample_pdf, "sample_policy.pdf", offline_settings(), lambda *a: None)
