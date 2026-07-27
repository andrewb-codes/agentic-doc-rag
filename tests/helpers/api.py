from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from agentic_rag.api.main import app
from agentic_rag.core.config import settings
from agentic_rag.rate_limit.deps import get_rate_limit_service
from agentic_rag.rate_limit.rules import RateLimitRule
from agentic_rag.rate_limit.service import RateLimitResult


class FakeRateLimitService:
    enabled = True

    def __init__(self, *, denied_scope: str | None = None) -> None:
        self.denied_scope = denied_scope
        self.calls: list[tuple[RateLimitRule, str]] = []

    async def hit(self, *, rule: RateLimitRule, key: str, cost: int = 1) -> RateLimitResult:
        self.calls.append((rule, key))
        allowed = rule.scope != self.denied_scope

        return RateLimitResult(
            allowed=allowed,
            limit=1,
            remaining=1 if allowed else 0,
            reset_at=123,
            retry_after=42,
        )


def internal_headers(
    *,
    telegram_user_id: int = 123456789,
    telegram_username: str | None = "andrew",
    internal_api_key: str = settings.internal_api_key,
) -> dict[str, str]:
    headers = {
        "X-Internal-API-Key": internal_api_key,
        "X-Telegram-User-Id": str(telegram_user_id),
    }

    if telegram_username is not None:
        headers["X-Telegram-Username"] = telegram_username

    return headers


@asynccontextmanager
async def app_client() -> AsyncIterator[AsyncClient]:
    try:
        async with (
            LifespanManager(app),
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client,
        ):
            yield client
    finally:
        app.dependency_overrides.clear()


@contextmanager
def override_rate_limit_service(service: FakeRateLimitService) -> Iterator[FakeRateLimitService]:
    app.dependency_overrides[get_rate_limit_service] = lambda: service

    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_rate_limit_service, None)
