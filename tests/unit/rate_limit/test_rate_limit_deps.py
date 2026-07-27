from typing import cast

import pytest
from fastapi import Response

from agentic_rag.rate_limit.deps import apply_rate_limit
from agentic_rag.rate_limit.exceptions import RateLimitExceededError
from agentic_rag.rate_limit.rules import RateLimitRule
from agentic_rag.rate_limit.scopes import RateLimitScope
from agentic_rag.rate_limit.service import RateLimitResult, RateLimitService

pytestmark = pytest.mark.no_db


class FakeRateLimitService:
    def __init__(self, *, result: RateLimitResult, enabled: bool = True) -> None:
        self.enabled = enabled
        self.result = result
        self.calls: list[tuple[RateLimitRule, str]] = []

    async def hit(self, *, rule: RateLimitRule, key: str) -> RateLimitResult:
        self.calls.append((rule, key))
        return self.result


async def test_apply_rate_limit_skips_disabled_service() -> None:
    rule = RateLimitRule(scope=RateLimitScope.DOCUMENT_ASK, limit="1 per minute")
    response = Response()
    service = FakeRateLimitService(
        enabled=False,
        result=RateLimitResult(allowed=False),
    )

    await apply_rate_limit(
        rule=rule,
        key_factory=lambda: "rate-limit:document_ask:user:1",
        response=response,
        service=cast(RateLimitService, cast(object, service)),
    )

    assert service.calls == []
    assert "X-RateLimit-Limit" not in response.headers


async def test_apply_rate_limit_sets_headers_when_request_is_allowed() -> None:
    rule = RateLimitRule(scope=RateLimitScope.DOCUMENT_ASK, limit="10 per hour")
    response = Response()
    service = FakeRateLimitService(
        result=RateLimitResult(
            allowed=True,
            limit=10,
            remaining=9,
            reset_at=1234567890,
        )
    )

    await apply_rate_limit(
        rule=rule,
        key_factory=lambda: "rate-limit:document_ask:user:1",
        response=response,
        service=cast(RateLimitService, cast(object, service)),
    )

    assert service.calls == [(rule, "rate-limit:document_ask:user:1")]
    assert response.headers["X-RateLimit-Limit"] == "10"
    assert response.headers["X-RateLimit-Remaining"] == "9"
    assert response.headers["X-RateLimit-Reset"] == "1234567890"


async def test_apply_rate_limit_raises_when_request_is_rejected() -> None:
    rule = RateLimitRule(scope=RateLimitScope.DOCUMENT_UPLOAD, limit="1 per hour")
    response = Response()
    service = FakeRateLimitService(
        result=RateLimitResult(
            allowed=False,
            limit=1,
            remaining=0,
            reset_at=1234567890,
            retry_after=60,
        )
    )

    with pytest.raises(RateLimitExceededError) as exc_info:
        await apply_rate_limit(
            rule=rule,
            key_factory=lambda: "rate-limit:document_upload:user:1",
            response=response,
            service=cast(RateLimitService, cast(object, service)),
        )

    assert exc_info.value.retry_after == 60
    assert response.headers["X-RateLimit-Limit"] == "1"
    assert response.headers["X-RateLimit-Remaining"] == "0"
    assert response.headers["X-RateLimit-Reset"] == "1234567890"
