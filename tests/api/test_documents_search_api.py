from unittest.mock import AsyncMock

from httpx import AsyncClient

from agentic_rag.api.deps import get_retrieval_service
from agentic_rag.api.main import app
from agentic_rag.db.session import AsyncSessionLocal
from agentic_rag.models import DocumentChunk
from tests.helpers import create_document, create_user, internal_headers


class FakeRetrievalService:
    def __init__(self, *, chunks: list[DocumentChunk]) -> None:
        self.search_user_chunks = AsyncMock(return_value=chunks)
        self.search_document_chunks = AsyncMock(return_value=chunks)


def override_retrieval_service(retrieval_service: FakeRetrievalService) -> None:
    async def get_fake_retrieval_service() -> FakeRetrievalService:
        return retrieval_service

    app.dependency_overrides[get_retrieval_service] = get_fake_retrieval_service


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
    override_retrieval_service(retrieval_service)

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


async def test_search_documents_uses_default_limit(client: AsyncClient) -> None:
    retrieval_service = FakeRetrievalService(chunks=[])
    override_retrieval_service(retrieval_service)

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


async def test_search_documents_rejects_empty_query(client: AsyncClient) -> None:
    response = await client.post(
        "/documents/search",
        headers=internal_headers(),
        json={"query": "", "limit": 5},
    )

    assert response.status_code == 422


async def test_search_documents_rejects_too_large_limit(client: AsyncClient) -> None:
    response = await client.post(
        "/documents/search",
        headers=internal_headers(),
        json={"query": "question", "limit": 21},
    )

    assert response.status_code == 422


async def test_search_document_returns_chunks_for_current_user_document(
    client: AsyncClient,
) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789, username="andrew")
        document = await create_document(session, owner_id=user.id, filename="manual.pdf")
        document_id = document.id
        await session.commit()

    chunk = DocumentChunk(
        id=10,
        document_id=document_id,
        page=3,
        chunk_index=0,
        text="chunk text",
        source="manual.pdf",
    )
    retrieval_service = FakeRetrievalService(chunks=[chunk])
    override_retrieval_service(retrieval_service)

    response = await client.post(
        f"/documents/{document_id}/search",
        headers=internal_headers(),
        json={"query": "question", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 10,
            "document_id": document_id,
            "page": 3,
            "chunk_index": 0,
            "text": "chunk text",
            "source": "manual.pdf",
        }
    ]
    retrieval_service.search_document_chunks.assert_awaited_once_with(
        query="question",
        owner_id=1,
        document_id=document_id,
        limit=5,
    )


async def test_search_foreign_document_returns_404(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        first_user = await create_user(session, telegram_user_id=111, username="first")
        document = await create_document(session, owner_id=first_user.id, filename="first.pdf")
        document_id = document.id
        await session.commit()

    retrieval_service = FakeRetrievalService(chunks=[])
    override_retrieval_service(retrieval_service)

    response = await client.post(
        f"/documents/{document_id}/search",
        headers=internal_headers(telegram_user_id=222, telegram_username="second"),
        json={"query": "question", "limit": 5},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "error.document.not_found"}
    retrieval_service.search_document_chunks.assert_not_awaited()
