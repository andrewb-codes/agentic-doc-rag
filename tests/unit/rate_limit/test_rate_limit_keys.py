import pytest

from agentic_rag.rate_limit.keys import build_user_key
from agentic_rag.rate_limit.scopes import RateLimitScope

pytestmark = pytest.mark.no_db


def test_user_key_contains_user_id_and_scope() -> None:
    key = build_user_key(scope=RateLimitScope.DOCUMENT_ASK, user_id=123)

    assert key == "rate-limit:document_ask:user:123"
