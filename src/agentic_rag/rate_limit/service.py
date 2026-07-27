import logging
import math
import time
from dataclasses import dataclass
from typing import Self

from limits import parse
from limits.aio.storage import RedisStorage
from limits.aio.strategies import SlidingWindowCounterRateLimiter
from limits.errors import StorageError

from agentic_rag.core.config import Settings
from agentic_rag.rate_limit.rules import RateLimitRule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None
    retry_after: int = 0


class RateLimitService:
    def __init__(
        self,
        *,
        enabled: bool,
        limiter: SlidingWindowCounterRateLimiter | None = None,
    ) -> None:
        self.enabled = enabled
        self.limiter = limiter

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        if not settings.rate_limit_enabled:
            return cls(enabled=False)

        if not settings.rate_limit_redis_url:
            raise RuntimeError("RATE_LIMIT_REDIS_URL is required when rate limiting is enabled")

        storage = RedisStorage(
            uri=settings.rate_limit_redis_url,
            wrap_exceptions=True,
            implementation=settings.rate_limit_redis_implementation,
            key_prefix=settings.rate_limit_key_prefix,
        )

        return cls(
            enabled=True,
            limiter=SlidingWindowCounterRateLimiter(storage),
        )

    async def check_storage(self) -> bool:
        if not self.enabled:
            return True

        if self.limiter is None:
            return False

        return await self.limiter.storage.check()

    async def hit(self, *, rule: RateLimitRule, key: str, cost: int = 1) -> RateLimitResult:
        if not self.enabled:
            return RateLimitResult(allowed=True)

        item = parse(rule.limit)

        if self.limiter is None:
            return self._handle_storage_failure(rule=rule, key=key)

        try:
            allowed = await self.limiter.hit(item, key, cost=cost)
            stats = await self.limiter.get_window_stats(item, key)
        except StorageError:
            logger.warning(
                "rate_limit.storage_error",
                extra={"scope": rule.scope, "failure_mode": rule.failure_mode},
                exc_info=True,
            )
            return self._handle_storage_failure(rule=rule, key=key)

        reset_at = math.ceil(stats.reset_time)
        retry_after = max(0, reset_at - math.ceil(time.time()))

        return RateLimitResult(
            allowed=allowed,
            limit=item.amount,
            remaining=max(0, stats.remaining),
            reset_at=reset_at,
            retry_after=retry_after,
        )

    @staticmethod
    def _handle_storage_failure(*, rule: RateLimitRule, key: str) -> RateLimitResult:
        if rule.failure_mode == "closed":
            logger.warning(
                "rate_limit.fail_closed",
                extra={"scope": rule.scope, "key": key},
            )
            return RateLimitResult(
                allowed=False,
                retry_after=1,
            )
        logger.warning(
            "rate_limit.fail_open",
            extra={"scope": rule.scope, "key": key},
        )
        return RateLimitResult(allowed=True)
