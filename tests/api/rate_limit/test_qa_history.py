from httpx import AsyncClient

from agentic_rag.rate_limit.rules import QA_HISTORY_READ_LIMIT
from tests.helpers import FakeRateLimitService, internal_headers, override_rate_limit_service


async def test_list_qa_history_applies_qa_history_read_rate_limit(
    client: AsyncClient,
) -> None:
    service = FakeRateLimitService()
    with override_rate_limit_service(service):
        response = await client.get("/qa-history", headers=internal_headers())

    assert response.status_code == 200

    rule, key = service.calls[0]
    assert rule.scope == QA_HISTORY_READ_LIMIT.scope
    assert key == "rate-limit:qa_history_read:user:1"


async def test_list_qa_history_returns_429_when_rate_limit_exceeded(
    client: AsyncClient,
) -> None:
    service = FakeRateLimitService(denied_scope=QA_HISTORY_READ_LIMIT.scope)
    with override_rate_limit_service(service):
        response = await client.get("/qa-history", headers=internal_headers())

    assert response.status_code == 429
    assert response.json() == {"detail": "error.rate_limit.exceeded"}
    assert response.headers["Retry-After"] == "42"
