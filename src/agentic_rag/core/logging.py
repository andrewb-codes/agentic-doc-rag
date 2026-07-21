import logging
import re
import sys
from collections.abc import Mapping
from typing import Any

import structlog

from agentic_rag.core.config import Settings

SENSITIVE_KEY_PARTS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "jwt",
    "database_url",
    "redis_url",
)
REDACTED = "[redacted]"

_URL_PASSWORD_PATTERN = re.compile(r"(?P<prefix>://[^:/@\s]+:)(?P<password>[^@\s]+)(?P<suffix>@)")
THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi")


def configure_logging(settings: Settings) -> None:
    log_level = _parse_log_level(settings.log_level, default=logging.INFO)
    sql_log_level = _parse_log_level(settings.sql_log_level, default=logging.WARNING)

    timestamper = structlog.processors.TimeStamper(
        fmt="iso",
        utc=True,
        key="timestamp",
    )
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_global_log_fields(settings),
        timestamper,
        redact_sensitive_values,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.typing.Processor
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    for logger_name in THIRD_PARTY_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(log_level)

    sqlalchemy_logger = logging.getLogger("sqlalchemy")
    sqlalchemy_logger.handlers.clear()
    sqlalchemy_logger.propagate = True
    sqlalchemy_logger.setLevel(sql_log_level)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def redact_sensitive_values(
    _: logging.Logger,
    __: str,
    event_dict: structlog.typing.EventDict,
) -> structlog.typing.EventDict:
    return {key: _redact_value(key, value) for key, value in event_dict.items()}


def add_global_log_fields(settings: Settings) -> structlog.typing.Processor:
    def processor(
        _: logging.Logger,
        __: str,
        event_dict: structlog.typing.EventDict,
    ) -> structlog.typing.EventDict:
        event_dict.setdefault("service", settings.app_name)
        event_dict.setdefault("environment", settings.environment)
        return event_dict

    return processor


def _redact_value(key: str, value: Any) -> Any:
    if any(part in key.lower() for part in SENSITIVE_KEY_PARTS):
        return REDACTED

    if isinstance(value, Mapping):
        return {
            nested_key: _redact_value(str(nested_key), nested_value)
            for nested_key, nested_value in value.items()
        }

    if isinstance(value, str):
        return _redact_url_password(value)

    return value


def _redact_url_password(value: str) -> str:
    return _URL_PASSWORD_PATTERN.sub(r"\g<prefix>[redacted]\g<suffix>", value)


def _parse_log_level(value: str, *, default: int) -> int:
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    return default
