from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentic_rag.services.embedding import FakeEmbeddingService, OpenAIEmbeddingService

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


async def test_openai_embedding_service_embeds_texts() -> None:
    client = AsyncMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1.0, 2.0, 3.0]),
                SimpleNamespace(embedding=[4.0, 5.0, 6.0]),
            ]
        )
    )

    service = OpenAIEmbeddingService(
        api_key="test-key",
        model="text-embedding-3-small",
        vector_size=3,
        client=client,
    )

    embeddings = await service.embed_texts(texts=["first", "second"])

    assert embeddings == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["first", "second"],
    )


async def test_openai_embedding_service_exposes_vector_size() -> None:
    service = OpenAIEmbeddingService(
        api_key="test-key",
        model="text-embedding-3-small",
        vector_size=1536,
        client=AsyncMock(),
    )
    assert service.vector_size == 1536
