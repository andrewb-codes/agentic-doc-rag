from httpx import AsyncClient

from agentic_rag.db.session import AsyncSessionLocal
from tests.helpers import create_document, create_qa_history, create_user, internal_headers


async def test_list_user_qa_history_returns_empty_list(client: AsyncClient) -> None:
    response = await client.get("/qa-history", headers=internal_headers())

    assert response.status_code == 200
    assert response.json() == []


async def test_list_user_qa_history_returns_current_user_history(
    client: AsyncClient,
) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789)
        document = await create_document(session, owner_id=user.id)
        await create_qa_history(
            session,
            user_id=user.id,
            document_id=document.id,
            question="When did Atlas start?",
            answer="Atlas started on March 14, 2025.",
        )
        await session.commit()

    response = await client.get("/qa-history", headers=internal_headers())
    body = response.json()

    assert response.status_code == 200
    assert len(body) == 1
    assert body[0]["id"] == 1
    assert body[0]["user_id"] == 1
    assert body[0]["document_id"] == 1
    assert body[0]["question"] == "When did Atlas start?"
    assert body[0]["answer"] == "Atlas started on March 14, 2025."
    assert body[0]["verification_verdict"] == "not_verified"
    assert body[0]["created_at"]


async def test_list_user_qa_history_returns_only_current_user_history(
    client: AsyncClient,
) -> None:
    async with AsyncSessionLocal() as session:
        first_user = await create_user(
            session,
            telegram_user_id=111,
            username="first",
        )
        second_user = await create_user(
            session,
            telegram_user_id=222,
            username="second",
        )
        await create_qa_history(
            session,
            user_id=first_user.id,
            question="First question?",
            answer="First answer.",
        )
        await create_qa_history(
            session,
            user_id=second_user.id,
            question="Second question?",
            answer="Second answer.",
        )
        await session.commit()

    response = await client.get(
        "/qa-history",
        headers=internal_headers(telegram_user_id=111, telegram_username="first"),
    )
    body = response.json()

    assert response.status_code == 200
    assert len(body) == 1
    assert body[0]["user_id"] == 1
    assert body[0]["question"] == "First question?"
    assert body[0]["answer"] == "First answer."


async def test_list_user_qa_history_returns_latest_items_first(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789)
        await create_qa_history(
            session,
            user_id=user.id,
            question="First question?",
            answer="First answer.",
        )
        await create_qa_history(
            session,
            user_id=user.id,
            question="Second question?",
            answer="Second answer.",
        )
        await session.commit()

    response = await client.get("/qa-history", headers=internal_headers())
    body = response.json()

    assert response.status_code == 200
    assert [item["question"] for item in body] == [
        "Second question?",
        "First question?",
    ]
