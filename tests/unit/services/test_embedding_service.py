import pytest

from agentic_rag.services.embedding import FakeEmbeddingService

pytestmark = pytest.mark.no_db


async def test_fake_embedding_service_returns_one_vector_per_text() -> None:
    service = FakeEmbeddingService(vector_size=3)

    embeddings = await service.embed_texts(texts=["first", "second"])

    assert embeddings == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]


async def test_fake_embedding_service_exposes_vector_size() -> None:
    service = FakeEmbeddingService(vector_size=1536)

    assert service.vector_size == 1536
