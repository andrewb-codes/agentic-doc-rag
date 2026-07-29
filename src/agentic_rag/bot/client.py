from dataclasses import dataclass
from typing import Any, cast

import httpx


@dataclass(frozen=True)
class TelegramUser:
    telegram_user_id: int
    username: str | None = None


class BackendClient:
    def __init__(
        self,
        *,
        base_url: str,
        internal_api_key: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_api_key = internal_api_key
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def list_documents(self, *, user: TelegramUser) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}/documents",
            headers=self._headers(user=user),
        )
        response.raise_for_status()
        return self._json_list(response)

    async def list_history(self, *, user: TelegramUser) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.base_url}/qa-history",
            headers=self._headers(user=user),
        )
        response.raise_for_status()
        return self._json_list(response)

    async def upload_document(
        self,
        *,
        user: TelegramUser,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/documents/upload",
            headers=self._headers(user=user),
            files={"file": (filename, content, "application/pdf")},
        )
        response.raise_for_status()
        return self._json_object(response)

    async def ask_documents(
        self,
        *,
        user: TelegramUser,
        question: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/documents/ask",
            headers=self._headers(user=user),
            json={"question": question, "limit": limit},
        )
        response.raise_for_status()
        return self._json_object(response)

    def _headers(self, *, user: TelegramUser) -> dict[str, str]:
        headers = {
            "X-Internal-API-Key": self.internal_api_key,
            "X-Telegram-User-Id": str(user.telegram_user_id),
        }

        if user.username is not None:
            headers["X-Telegram-Username"] = user.username

        return headers

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        return cast(dict[str, Any], response.json())

    @staticmethod
    def _json_list(response: httpx.Response) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], response.json())
