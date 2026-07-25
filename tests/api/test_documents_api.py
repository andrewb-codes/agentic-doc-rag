import pytest
from httpx import AsyncClient

from agentic_rag.core.config import settings
from tests.api.helpers import internal_headers


@pytest.mark.no_db
async def test_create_document_without_internal_api_key_returns_401(client: AsyncClient) -> None:
    response = await client.post(
        "/documents",
        headers={"X-Telegram-User-Id": "123456789"},
        json={"filename": "manual.pdf"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "error.auth.unauthorized"}


@pytest.mark.no_db
async def test_create_document_without_telegram_user_id_returns_401(client: AsyncClient) -> None:
    response = await client.post(
        "/documents",
        headers={"X-Internal-API-Key": settings.internal_api_key},
        json={"filename": "manual.pdf"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "error.auth.unauthorized"}


async def test_create_document_creates_telegram_user_automatically(client: AsyncClient) -> None:
    response = await client.post(
        "/documents",
        headers=internal_headers(),
        json={"filename": "manual.pdf"},
    )

    body = response.json()

    assert response.status_code == 201
    assert body["id"] == 1
    assert body["owner_id"] == 1
    assert body["filename"] == "manual.pdf"
    assert body["status"] == "pending"
    assert body["page_count"] is None
    assert body["chunk_count"] is None
    assert body["created_at"]


async def test_list_documents_returns_only_current_telegram_user_documents(
    client: AsyncClient,
) -> None:
    first_user_headers = internal_headers(telegram_user_id=111, telegram_username="first")
    second_user_headers = internal_headers(telegram_user_id=222, telegram_username="second")

    first_response = await client.post(
        "/documents",
        headers=first_user_headers,
        json={"filename": "first.pdf"},
    )
    await client.post(
        "/documents",
        headers=second_user_headers,
        json={"filename": "second.pdf"},
    )

    response = await client.get("/documents", headers=first_user_headers)
    body = response.json()

    assert response.status_code == 200
    assert body == [first_response.json()]


async def test_get_document_returns_current_user_document(client: AsyncClient) -> None:
    create_response = await client.post(
        "/documents",
        headers=internal_headers(),
        json={"filename": "manual.pdf"},
    )
    document_id = create_response.json()["id"]

    response = await client.get(f"/documents/{document_id}", headers=internal_headers())

    assert response.status_code == 200
    assert response.json() == create_response.json()


async def test_get_foreign_document_returns_404(client: AsyncClient) -> None:
    create_response = await client.post(
        "/documents",
        headers=internal_headers(telegram_user_id=111, telegram_username="first"),
        json={"filename": "first.pdf"},
    )
    document_id = create_response.json()["id"]

    response = await client.get(
        f"/documents/{document_id}",
        headers=internal_headers(telegram_user_id=222, telegram_username="second"),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "error.document.not_found"}
