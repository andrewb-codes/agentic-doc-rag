from unittest.mock import AsyncMock

from httpx import AsyncClient

from agentic_rag.api.deps import get_retrieval_service
from agentic_rag.api.main import app
from agentic_rag.models import DocumentChunk
from tests.api.helpers import internal_headers


class FakeRetrievalService:
    def __init__(self, *, chunks: list[DocumentChunk]) -> None:
        self.search_user_chunks = AsyncMock(return_value=chunks)


async def test_search_documents_returns_relevant_chunks_for_current_user(
    client: AsyncClient,
) -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="chunk text",
        source="manual.pdf",
    )
    retrieval_service = FakeRetrievalService(chunks=[chunk])

    async def get_fake_retrieval_service() -> FakeRetrievalService:
        return retrieval_service

    app.dependency_overrides[get_retrieval_service] = get_fake_retrieval_service

    response = await client.post(
        "/documents/search",
        headers=internal_headers(),
        json={"query": "question", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 10,
            "document_id": 20,
            "page": 3,
            "chunk_index": 0,
            "text": "chunk text",
            "source": "manual.pdf",
        }
    ]
    retrieval_service.search_user_chunks.assert_awaited_once_with(
        query="question",
        owner_id=1,
        limit=5,
    )


async def test_search_documents_uses_default_limit(
    client: AsyncClient,
) -> None:
    retrieval_service = FakeRetrievalService(chunks=[])

    async def get_fake_retrieval_service() -> FakeRetrievalService:
        return retrieval_service

    app.dependency_overrides[get_retrieval_service] = get_fake_retrieval_service

    response = await client.post(
        "/documents/search",
        headers=internal_headers(),
        json={"query": "question"},
    )

    assert response.status_code == 200
    assert response.json() == []
    retrieval_service.search_user_chunks.assert_awaited_once_with(
        query="question",
        owner_id=1,
        limit=5,
    )


async def test_search_documents_rejects_empty_query(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/documents/search",
        headers=internal_headers(),
        json={"query": "", "limit": 5},
    )

    assert response.status_code == 422


async def test_search_documents_rejects_too_large_limit(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/documents/search",
        headers=internal_headers(),
        json={"query": "question", "limit": 21},
    )

    assert response.status_code == 422
