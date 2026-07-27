from unittest.mock import AsyncMock

from httpx import AsyncClient
from sqlalchemy import select

from agentic_rag.api.deps import get_answer_service, get_chat_service, get_retrieval_service
from agentic_rag.api.main import app
from agentic_rag.db.session import AsyncSessionLocal
from agentic_rag.models import DocumentChunk, QAHistory, VerificationVerdict
from agentic_rag.services.answer import AnswerResult
from tests.helpers import create_document, create_user, internal_headers


class FakeAnswerService:
    def __init__(self, *, result: AnswerResult) -> None:
        self.answer_user_question = AsyncMock(return_value=result)
        self.answer_document_question = AsyncMock(return_value=result)


class FakeRetrievalService:
    def __init__(self, *, chunks: list[DocumentChunk]) -> None:
        self.search_user_chunks = AsyncMock(return_value=chunks)
        self.search_document_chunks = AsyncMock(return_value=chunks)


class FakeChatService:
    def __init__(
        self,
        *,
        answer: str,
        verification_verdict: str = "supported",
    ) -> None:
        self.answer_question = AsyncMock(return_value=answer)
        self.verify_answer = AsyncMock(return_value=verification_verdict)


def override_answer_service(answer_service: FakeAnswerService) -> None:
    async def get_fake_answer_service() -> FakeAnswerService:
        return answer_service

    app.dependency_overrides[get_answer_service] = get_fake_answer_service


def override_answer_dependencies(
    *,
    retrieval_service: FakeRetrievalService,
    chat_service: FakeChatService,
) -> None:
    async def get_fake_retrieval_service() -> FakeRetrievalService:
        return retrieval_service

    async def get_fake_chat_service() -> FakeChatService:
        return chat_service

    app.dependency_overrides[get_retrieval_service] = get_fake_retrieval_service
    app.dependency_overrides[get_chat_service] = get_fake_chat_service


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
        result=AnswerResult(
            answer="Atlas started on March 14, 2025.",
            chunks=[chunk],
            verification_verdict=VerificationVerdict.SUPPORTED,
        )
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
        "verification_verdict": "supported",
    }
    answer_service.answer_user_question.assert_awaited_once_with(
        question="When did Atlas start?",
        owner_id=1,
        limit=5,
    )


async def test_ask_document_returns_answer_for_current_user_document(client: AsyncClient) -> None:
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
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    answer_service = FakeAnswerService(
        result=AnswerResult(
            answer="Atlas started on March 14, 2025.",
            chunks=[chunk],
            verification_verdict=VerificationVerdict.SUPPORTED,
        )
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


async def test_ask_documents_saves_qa_history(client: AsyncClient) -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    retrieval_service = FakeRetrievalService(chunks=[chunk])
    chat_service = FakeChatService(
        answer="Atlas started on March 14, 2025.",
        verification_verdict="supported",
    )
    override_answer_dependencies(
        retrieval_service=retrieval_service,
        chat_service=chat_service,
    )

    response = await client.post(
        "/documents/ask",
        headers=internal_headers(),
        json={"question": "When did Atlas start?", "limit": 5},
    )

    async with AsyncSessionLocal() as session:
        history_item = await session.scalar(select(QAHistory))

    assert response.status_code == 200
    assert history_item is not None
    assert history_item.user_id == 1
    assert history_item.document_id is None
    assert history_item.question == "When did Atlas start?"
    assert history_item.answer == "Atlas started on March 14, 2025."
    assert history_item.verification_verdict == VerificationVerdict.SUPPORTED


async def test_ask_document_saves_qa_history_with_document_id(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789)
        document = await create_document(session, owner_id=user.id)
        document_id = document.id
        await session.commit()

    chunk = DocumentChunk(
        id=10,
        document_id=document_id,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    retrieval_service = FakeRetrievalService(chunks=[chunk])
    chat_service = FakeChatService(
        answer="Atlas started on March 14, 2025.",
        verification_verdict="supported",
    )
    override_answer_dependencies(
        retrieval_service=retrieval_service,
        chat_service=chat_service,
    )

    response = await client.post(
        f"/documents/{document_id}/ask",
        headers=internal_headers(),
        json={"question": "When did Atlas start?", "limit": 5},
    )

    async with AsyncSessionLocal() as session:
        history_item = await session.scalar(select(QAHistory))

    assert response.status_code == 200
    assert history_item is not None
    assert history_item.user_id == 1
    assert history_item.document_id == document_id
    assert history_item.question == "When did Atlas start?"
    assert history_item.answer == "Atlas started on March 14, 2025."
    assert history_item.verification_verdict == VerificationVerdict.SUPPORTED


async def test_ask_documents_uses_default_limit(client: AsyncClient) -> None:
    answer_service = FakeAnswerService(
        result=AnswerResult(
            answer="No answer.",
            chunks=[],
            verification_verdict=VerificationVerdict.UNSUPPORTED,
        )
    )
    override_answer_service(answer_service)

    response = await client.post(
        "/documents/ask",
        headers=internal_headers(),
        json={"question": "Unknown?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "No answer.",
        "chunks": [],
        "verification_verdict": "unsupported",
    }
    answer_service.answer_user_question.assert_awaited_once_with(
        question="Unknown?",
        owner_id=1,
        limit=5,
    )


async def test_ask_foreign_document_returns_404(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        first_user = await create_user(session, telegram_user_id=111, username="first")
        document = await create_document(session, owner_id=first_user.id, filename="first.pdf")
        document_id = document.id
        await session.commit()

    answer_service = FakeAnswerService(
        result=AnswerResult(
            answer="",
            chunks=[],
            verification_verdict=VerificationVerdict.UNSUPPORTED,
        )
    )
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
