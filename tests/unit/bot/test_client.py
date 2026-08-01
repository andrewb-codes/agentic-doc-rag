from datetime import datetime

import httpx
import pytest

from agentic_rag.bot.client import (
    BackendClient,
    BackendResponseValidationError,
    BotAskResponse,
    BotDocumentChunkResponse,
    BotDocumentResponse,
    BotQAHistoryResponse,
    BotVerificationResult,
    TelegramUser,
)
from agentic_rag.core.enums import AnswerStatus, DocumentStatus, VerificationVerdict


def document_payload() -> dict[str, object]:
    return {
        "id": 10,
        "owner_id": 20,
        "filename": "report.pdf",
        "status": "processed",
        "page_count": 3,
        "chunk_count": 7,
        "created_at": "2026-07-31T12:00:00",
    }


def build_backend_client(handler: httpx.MockTransport) -> BackendClient:
    return BackendClient(
        base_url="http://backend.test",
        internal_api_key="secret",
        client=httpx.AsyncClient(transport=handler),
    )


async def test_backend_client_validates_success_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Internal-API-Key"] == "secret"
        assert request.headers["X-Telegram-User-Id"] == "123"
        assert request.headers["X-Telegram-Username"] == "andrew"

        if request.method == "GET" and request.url.path == "/documents":
            return httpx.Response(200, json=[document_payload()])

        if request.method == "GET" and request.url.path == "/qa-history":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 30,
                        "user_id": 20,
                        "document_id": 10,
                        "question": "When?",
                        "answer": "On July 31.",
                        "verification_result": {
                            "verdict": "supported",
                            "unsupported_claims": [],
                            "missing_information": [],
                            "confidence": 1.0,
                        },
                        "created_at": "2026-07-31T12:05:00",
                    }
                ],
            )

        if request.method == "POST" and request.url.path == "/documents/ask":
            return httpx.Response(
                200,
                json={
                    "answer": "On July 31.",
                    "answer_status": "answered",
                    "chunks": [
                        {
                            "id": 40,
                            "document_id": 10,
                            "page": 1,
                            "chunk_index": 0,
                            "text": "The event is on July 31.",
                            "source": "report.pdf",
                        }
                    ],
                    "verification_result": {
                        "verdict": "supported",
                        "unsupported_claims": [],
                        "missing_information": [],
                        "confidence": 1.0,
                    },
                },
            )

        return httpx.Response(404)

    backend = build_backend_client(httpx.MockTransport(handler))

    try:
        user = TelegramUser(telegram_user_id=123, username="andrew")

        documents = await backend.list_documents(user=user)
        history = await backend.list_history(user=user)
        answer = await backend.ask_documents(user=user, question="When?")
    finally:
        await backend.close()

    assert documents == [
        BotDocumentResponse(
            id=10,
            owner_id=20,
            filename="report.pdf",
            status=DocumentStatus.PROCESSED,
            page_count=3,
            chunk_count=7,
            created_at=datetime(2026, 7, 31, 12, 0),
        )
    ]
    assert history == [
        BotQAHistoryResponse(
            id=30,
            user_id=20,
            document_id=10,
            question="When?",
            answer="On July 31.",
            verification_result=BotVerificationResult(
                verdict=VerificationVerdict.SUPPORTED,
                unsupported_claims=[],
                missing_information=[],
                confidence=1.0,
            ),
            created_at=datetime(2026, 7, 31, 12, 5),
        )
    ]
    assert answer == BotAskResponse(
        answer="On July 31.",
        answer_status=AnswerStatus.ANSWERED,
        chunks=[
            BotDocumentChunkResponse(
                id=40,
                document_id=10,
                page=1,
                chunk_index=0,
                text="The event is on July 31.",
                source="report.pdf",
            )
        ],
        verification_result=BotVerificationResult(
            verdict=VerificationVerdict.SUPPORTED,
            unsupported_claims=[],
            missing_information=[],
            confidence=1.0,
        ),
    )


async def test_backend_client_raises_for_invalid_response_payload() -> None:
    backend = build_backend_client(
        httpx.MockTransport(lambda _request: httpx.Response(200, json=[{"id": 10}]))
    )

    try:
        with pytest.raises(BackendResponseValidationError):
            await backend.list_documents(user=TelegramUser(telegram_user_id=123))
    finally:
        await backend.close()
