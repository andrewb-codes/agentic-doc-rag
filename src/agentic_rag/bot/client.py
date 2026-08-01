from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

from agentic_rag.core.enums import AnswerStatus, DocumentStatus, VerificationVerdict

ResponseT = TypeVar("ResponseT")


@dataclass(frozen=True)
class TelegramUser:
    telegram_user_id: int
    username: str | None = None


class BotDocumentResponse(BaseModel):
    id: int
    owner_id: int
    filename: str
    status: DocumentStatus
    page_count: int | None
    chunk_count: int | None
    created_at: datetime


class BotDocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    page: int
    chunk_index: int
    text: str
    source: str


class BotUnsupportedClaim(BaseModel):
    claim: str
    reason: str


class BotVerificationResult(BaseModel):
    verdict: VerificationVerdict | None = None
    unsupported_claims: list[BotUnsupportedClaim]
    missing_information: list[str]
    confidence: float | None


class BotAskResponse(BaseModel):
    answer: str
    answer_status: AnswerStatus
    chunks: list[BotDocumentChunkResponse]
    verification_result: BotVerificationResult


class BotQAHistoryResponse(BaseModel):
    id: int
    user_id: int
    document_id: int | None
    question: str
    answer: str
    verification_result: BotVerificationResult
    created_at: datetime


class BackendResponseValidationError(Exception):
    pass


DOCUMENT_RESPONSE_ADAPTER: TypeAdapter[BotDocumentResponse] = TypeAdapter(BotDocumentResponse)
DOCUMENTS_RESPONSE_ADAPTER: TypeAdapter[list[BotDocumentResponse]] = TypeAdapter(
    list[BotDocumentResponse]
)
ASK_RESPONSE_ADAPTER: TypeAdapter[BotAskResponse] = TypeAdapter(BotAskResponse)
QA_HISTORY_RESPONSE_ADAPTER: TypeAdapter[list[BotQAHistoryResponse]] = TypeAdapter(
    list[BotQAHistoryResponse]
)


class BackendClient:
    def __init__(
        self,
        *,
        base_url: str,
        internal_api_key: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_api_key = internal_api_key
        self.client = client if client is not None else httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def list_documents(self, *, user: TelegramUser) -> list[BotDocumentResponse]:
        response = await self.client.get(
            f"{self.base_url}/documents",
            headers=self._headers(user=user),
        )
        response.raise_for_status()
        return self._validate_response(response, DOCUMENTS_RESPONSE_ADAPTER)

    async def list_history(self, *, user: TelegramUser) -> list[BotQAHistoryResponse]:
        response = await self.client.get(
            f"{self.base_url}/qa-history",
            headers=self._headers(user=user),
        )
        response.raise_for_status()
        return self._validate_response(response, QA_HISTORY_RESPONSE_ADAPTER)

    async def upload_document(
        self,
        *,
        user: TelegramUser,
        filename: str,
        content: bytes,
    ) -> BotDocumentResponse:
        response = await self.client.post(
            f"{self.base_url}/documents/upload",
            headers=self._headers(user=user),
            files={"file": (filename, content, "application/pdf")},
        )
        response.raise_for_status()
        return self._validate_response(response, DOCUMENT_RESPONSE_ADAPTER)

    async def ask_documents(
        self,
        *,
        user: TelegramUser,
        question: str,
        limit: int = 5,
    ) -> BotAskResponse:
        response = await self.client.post(
            f"{self.base_url}/documents/ask",
            headers=self._headers(user=user),
            json={"question": question, "limit": limit},
        )
        response.raise_for_status()
        return self._validate_response(response, ASK_RESPONSE_ADAPTER)

    async def ask_document(
        self,
        *,
        user: TelegramUser,
        document_id: int,
        question: str,
        limit: int = 5,
    ) -> BotAskResponse:
        response = await self.client.post(
            f"{self.base_url}/documents/{document_id}/ask",
            headers=self._headers(user=user),
            json={"question": question, "limit": limit},
        )
        response.raise_for_status()
        return self._validate_response(response, ASK_RESPONSE_ADAPTER)

    async def delete_document(self, *, user: TelegramUser, document_id: int) -> None:
        response = await self.client.delete(
            f"{self.base_url}/documents/{document_id}",
            headers=self._headers(user=user),
        )
        response.raise_for_status()

    def _headers(self, *, user: TelegramUser) -> dict[str, str]:
        headers = {
            "X-Internal-API-Key": self.internal_api_key,
            "X-Telegram-User-Id": str(user.telegram_user_id),
        }

        if user.username is not None:
            headers["X-Telegram-Username"] = user.username

        return headers

    @staticmethod
    def _validate_response(response: httpx.Response, adapter: TypeAdapter[ResponseT]) -> ResponseT:
        try:
            return adapter.validate_python(response.json())
        except ValueError as exc:
            raise BackendResponseValidationError("backend returned invalid response") from exc
