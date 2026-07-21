import re
import time
from collections.abc import Awaitable, Callable
from typing import Final
from uuid import uuid4

import structlog
from fastapi import Request, Response

REQUEST_ID_HEADER: Final = "X-Request-ID"
MAX_REQUEST_ID_LENGTH: Final = 64
REQUEST_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9._-]+$")

logger = structlog.get_logger(__name__)


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = _get_request_id(request.headers.get(REQUEST_ID_HEADER))
    started_at = time.perf_counter()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        http_method=request.method,
        http_path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            status_code=500,
            duration_ms=_duration_ms(started_at),
        )
        structlog.contextvars.clear_contextvars()
        raise

    response.headers[REQUEST_ID_HEADER] = request_id
    _log_request_completed(
        status_code=response.status_code,
        duration_ms=_duration_ms(started_at),
    )
    structlog.contextvars.clear_contextvars()

    return response


def _get_request_id(header_value: str | None) -> str:
    if (
        header_value
        and len(header_value) <= MAX_REQUEST_ID_LENGTH
        and REQUEST_ID_PATTERN.fullmatch(header_value)
    ):
        return header_value

    return str(uuid4())


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def _log_request_completed(*, status_code: int, duration_ms: float) -> None:
    log_method = logger.info

    if status_code >= 500:
        log_method = logger.error
    elif status_code >= 400:
        log_method = logger.warning

    log_method(
        "request_completed",
        status_code=status_code,
        duration_ms=duration_ms,
    )
