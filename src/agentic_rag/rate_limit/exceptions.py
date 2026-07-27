from agentic_rag.core.exceptions import AppError


class RateLimitExceededError(AppError):
    status_code = 429
    detail = "error.rate_limit.exceeded"

    def __init__(self, *, retry_after: int) -> None:
        self.retry_after = retry_after
