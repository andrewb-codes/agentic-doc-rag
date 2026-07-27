import pytest
from httpx import AsyncClient

from agentic_rag.core.config import settings
from agentic_rag.db.session import AsyncSessionLocal
from agentic_rag.models import DocumentStatus
from tests.helpers import create_document, create_user, internal_headers


@pytest.mark.no_db
async def test_list_documents_without_internal_api_key_returns_401(client: AsyncClient) -> None:
    response = await client.get(
        "/documents",
        headers={"X-Telegram-User-Id": "123456789"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "error.auth.unauthorized"}


@pytest.mark.no_db
async def test_list_documents_without_telegram_user_id_returns_401(client: AsyncClient) -> None:
    response = await client.get(
        "/documents",
        headers={"X-Internal-API-Key": settings.internal_api_key},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "error.auth.unauthorized"}


async def test_list_documents_returns_only_current_telegram_user_documents(
    client: AsyncClient,
) -> None:
    async with AsyncSessionLocal() as session:
        first_user = await create_user(session, telegram_user_id=111, username="first")
        second_user = await create_user(session, telegram_user_id=222, username="second")
        first_document = await create_document(
            session,
            owner_id=first_user.id,
            filename="first.pdf",
            status=DocumentStatus.PENDING,
        )
        await create_document(
            session,
            owner_id=second_user.id,
            filename="second.pdf",
            status=DocumentStatus.PENDING,
        )
        await session.commit()

    response = await client.get(
        "/documents",
        headers=internal_headers(telegram_user_id=111, telegram_username="first"),
    )
    body = response.json()

    assert response.status_code == 200
    assert len(body) == 1
    assert body[0]["id"] == first_document.id
    assert body[0]["owner_id"] == first_user.id
    assert body[0]["filename"] == "first.pdf"
    assert body[0]["status"] == "pending"
    assert body[0]["page_count"] is None
    assert body[0]["chunk_count"] is None
    assert body[0]["created_at"]


async def test_get_document_returns_current_user_document(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789, username="andrew")
        document = await create_document(
            session,
            owner_id=user.id,
            filename="manual.pdf",
            status=DocumentStatus.PENDING,
        )
        document_id = document.id
        await session.commit()

    response = await client.get(f"/documents/{document_id}", headers=internal_headers())
    body = response.json()

    assert response.status_code == 200
    assert body["id"] == document_id
    assert body["owner_id"] == user.id
    assert body["filename"] == "manual.pdf"
    assert body["status"] == "pending"
    assert body["page_count"] is None
    assert body["chunk_count"] is None
    assert body["created_at"]


async def test_get_foreign_document_returns_404(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        first_user = await create_user(session, telegram_user_id=111, username="first")
        document = await create_document(session, owner_id=first_user.id, filename="first.pdf")
        document_id = document.id
        await session.commit()

    response = await client.get(
        f"/documents/{document_id}",
        headers=internal_headers(telegram_user_id=222, telegram_username="second"),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "error.document.not_found"}
