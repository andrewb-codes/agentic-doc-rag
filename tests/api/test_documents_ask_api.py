from unittest.mock import AsyncMock

from httpx import AsyncClient

from agentic_rag.api.deps import get_answer_service
from agentic_rag.api.main import app
from agentic_rag.models import DocumentChunk
from agentic_rag.services.answer import AnswerResult
from tests.api.helpers import internal_headers


class FakeAnswerService:
    def __init__(self, *, result: AnswerResult) -> None:
        self.answer_user_question = AsyncMock(return_value=result)
        self.answer_document_question = AsyncMock(return_value=result)


def override_answer_service(answer_service: FakeAnswerService) -> None:
    async def get_fake_answer_service() -> FakeAnswerService:
        return answer_service

    app.dependency_overrides[get_answer_service] = get_fake_answer_service


async def test_ask_documents_returns_answer_for_current_user(client: AsyncClient) -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    answer_service = FakeAnswerService(
        result=AnswerResult(answer="Atlas started on March 14, 2025.", chunks=[chunk])
    )
    override_answer_service(answer_service)

    response = await client.post(
        "/documents/ask",
        headers=internal_headers(),
        json={"question": "When did Atlas start?", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Atlas started on March 14, 2025.",
        "chunks": [
            {
                "id": 10,
                "document_id": 20,
                "page": 3,
                "chunk_index": 0,
                "text": "Project Atlas started on March 14, 2025.",
                "source": "manual.pdf",
            }
        ],
    }
    answer_service.answer_user_question.assert_awaited_once_with(
        question="When did Atlas start?",
        owner_id=1,
        limit=5,
    )


async def test_ask_document_returns_answer_for_current_user_document(client: AsyncClient) -> None:
    create_response = await client.post(
        "/documents",
        headers=internal_headers(),
        json={"filename": "manual.pdf"},
    )
    document_id = create_response.json()["id"]

    chunk = DocumentChunk(
        id=10,
        document_id=document_id,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    answer_service = FakeAnswerService(
        result=AnswerResult(answer="Atlas started on March 14, 2025.", chunks=[chunk])
    )
    override_answer_service(answer_service)

    response = await client.post(
        f"/documents/{document_id}/ask",
        headers=internal_headers(),
        json={"question": "When did Atlas start?", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Atlas started on March 14, 2025."
    answer_service.answer_document_question.assert_awaited_once_with(
        question="When did Atlas start?",
        owner_id=1,
        document_id=document_id,
        limit=5,
    )


async def test_ask_documents_uses_default_limit(client: AsyncClient) -> None:
    answer_service = FakeAnswerService(result=AnswerResult(answer="No answer.", chunks=[]))
    override_answer_service(answer_service)

    response = await client.post(
        "/documents/ask",
        headers=internal_headers(),
        json={"question": "Unknown?"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "No answer.", "chunks": []}
    answer_service.answer_user_question.assert_awaited_once_with(
        question="Unknown?",
        owner_id=1,
        limit=5,
    )


async def test_ask_foreign_document_returns_404(client: AsyncClient) -> None:
    create_response = await client.post(
        "/documents",
        headers=internal_headers(telegram_user_id=111, telegram_username="first"),
        json={"filename": "first.pdf"},
    )
    document_id = create_response.json()["id"]

    answer_service = FakeAnswerService(result=AnswerResult(answer="", chunks=[]))
    override_answer_service(answer_service)

    response = await client.post(
        f"/documents/{document_id}/ask",
        headers=internal_headers(telegram_user_id=222, telegram_username="second"),
        json={"question": "Question?", "limit": 5},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "error.document.not_found"}
    answer_service.answer_document_question.assert_not_awaited()


async def test_ask_documents_rejects_empty_question(client: AsyncClient) -> None:
    response = await client.post(
        "/documents/ask",
        headers=internal_headers(),
        json={"question": "", "limit": 5},
    )

    assert response.status_code == 422


async def test_ask_documents_rejects_too_large_limit(client: AsyncClient) -> None:
    response = await client.post(
        "/documents/ask",
        headers=internal_headers(),
        json={"question": "Question?", "limit": 21},
    )

    assert response.status_code == 422
