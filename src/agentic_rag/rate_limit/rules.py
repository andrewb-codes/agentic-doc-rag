from dataclasses import dataclass
from typing import Literal

from agentic_rag.rate_limit.scopes import RateLimitScope

RateLimitFailureMode = Literal["open", "closed"]


@dataclass(frozen=True)
class RateLimitRule:
    scope: RateLimitScope
    limit: str
    failure_mode: RateLimitFailureMode = "open"


DOCUMENT_UPLOAD_LIMIT = RateLimitRule(
    scope=RateLimitScope.DOCUMENT_UPLOAD,
    limit="10 per hour",
)

DOCUMENT_SEARCH_LIMIT = RateLimitRule(
    scope=RateLimitScope.DOCUMENT_SEARCH,
    limit="120 per hour",
)

DOCUMENT_ASK_LIMIT = RateLimitRule(
    scope=RateLimitScope.DOCUMENT_ASK,
    limit="60 per hour",
)

QA_HISTORY_READ_LIMIT = RateLimitRule(
    scope=RateLimitScope.QA_HISTORY_READ,
    limit="120 per hour",
)
