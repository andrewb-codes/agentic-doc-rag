from unittest.mock import AsyncMock

from httpx import AsyncClient

from agentic_rag.api.deps import get_answer_service, get_retrieval_service
from agentic_rag.api.main import app
from agentic_rag.db.session import AsyncSessionLocal
from agentic_rag.models import VerificationVerdict
from agentic_rag.rate_limit.rules import (
    DOCUMENT_ASK_LIMIT,
    DOCUMENT_SEARCH_LIMIT,
    DOCUMENT_UPLOAD_LIMIT,
)
from agentic_rag.services.answer import AnswerResult
from tests.helpers import (
    FakeRateLimitService,
    create_document,
    create_user,
    internal_headers,
    override_rate_limit_service,
)


class FakeAnswerService:
    def __init__(self) -> None:
        self.answer_user_question = AsyncMock(
            return_value=AnswerResult(
                answer="No answer.",
                chunks=[],
                verification_verdict=VerificationVerdict.UNSUPPORTED,
            )
        )
        self.answer_document_question = AsyncMock(
            return_value=AnswerResult(
                answer="No answer.",
                chunks=[],
                verification_verdict=VerificationVerdict.UNSUPPORTED,
            )
        )


class FakeRetrievalService:
    def __init__(self) -> None:
        self.search_user_chunks = AsyncMock(return_value=[])
        self.search_document_chunks = AsyncMock(return_value=[])


def override_answer_service(answer_service: FakeAnswerService) -> None:
    async def get_fake_answer_service() -> FakeAnswerService:
        return answer_service

    app.dependency_overrides[get_answer_service] = get_fake_answer_service


def override_retrieval_service(retrieval_service: FakeRetrievalService) -> None:
    async def get_fake_retrieval_service() -> FakeRetrievalService:
        return retrieval_service

    app.dependency_overrides[get_retrieval_service] = get_fake_retrieval_service


async def test_ask_documents_applies_document_ask_rate_limit(client: AsyncClient) -> None:
    override_answer_service(FakeAnswerService())

    service = FakeRateLimitService()
    with override_rate_limit_service(service):
        response = await client.post(
            "/documents/ask",
            headers=internal_headers(),
            json={"question": "Question?"},
        )

    assert response.status_code == 200

    rule, key = service.calls[0]
    assert rule.scope == DOCUMENT_ASK_LIMIT.scope
    assert key == "rate-limit:document_ask:user:1"


async def test_ask_documents_returns_429_when_rate_limit_exceeded(
    client: AsyncClient,
) -> None:
    service = FakeRateLimitService(denied_scope=DOCUMENT_ASK_LIMIT.scope)
    with override_rate_limit_service(service):
        response = await client.post(
            "/documents/ask",
            headers=internal_headers(),
            json={"question": "Question?"},
        )

    assert response.status_code == 429
    assert response.json() == {"detail": "error.rate_limit.exceeded"}
    assert response.headers["Retry-After"] == "42"


async def test_ask_document_applies_document_ask_rate_limit(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789, username="andrew")
        document = await create_document(session, owner_id=user.id)
        document_id = document.id
        await session.commit()

    override_answer_service(FakeAnswerService())

    service = FakeRateLimitService()
    with override_rate_limit_service(service):
        response = await client.post(
            f"/documents/{document_id}/ask",
            headers=internal_headers(),
            json={"question": "Question?"},
        )

    assert response.status_code == 200

    rule, key = service.calls[0]
    assert rule.scope == DOCUMENT_ASK_LIMIT.scope
    assert key == "rate-limit:document_ask:user:1"


async def test_search_documents_applies_document_search_rate_limit(client: AsyncClient) -> None:
    override_retrieval_service(FakeRetrievalService())

    service = FakeRateLimitService()
    with override_rate_limit_service(service):
        response = await client.post(
            "/documents/search",
            headers=internal_headers(),
            json={"query": "Question?"},
        )

    assert response.status_code == 200

    rule, key = service.calls[0]
    assert rule.scope == DOCUMENT_SEARCH_LIMIT.scope
    assert key == "rate-limit:document_search:user:1"


async def test_search_document_applies_document_search_rate_limit(client: AsyncClient) -> None:
    async with AsyncSessionLocal() as session:
        user = await create_user(session, telegram_user_id=123456789, username="andrew")
        document = await create_document(session, owner_id=user.id)
        document_id = document.id
        await session.commit()

    override_retrieval_service(FakeRetrievalService())

    service = FakeRateLimitService()
    with override_rate_limit_service(service):
        response = await client.post(
            f"/documents/{document_id}/search",
            headers=internal_headers(),
            json={"query": "Question?"},
        )

    assert response.status_code == 200

    rule, key = service.calls[0]
    assert rule.scope == DOCUMENT_SEARCH_LIMIT.scope
    assert key == "rate-limit:document_search:user:1"


async def test_upload_document_returns_429_when_rate_limit_exceeded(client: AsyncClient) -> None:
    service = FakeRateLimitService(denied_scope=DOCUMENT_UPLOAD_LIMIT.scope)
    with override_rate_limit_service(service):
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("manual.pdf", b"%PDF-1.4\n", "application/pdf")},
        )

    assert response.status_code == 429
    assert response.json() == {"detail": "error.rate_limit.exceeded"}
    assert response.headers["Retry-After"] == "42"

    rule, key = service.calls[0]
    assert rule.scope == DOCUMENT_UPLOAD_LIMIT.scope
    assert key == "rate-limit:document_upload:user:1"
