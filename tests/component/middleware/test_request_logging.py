import json
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field

from agentic_rag.core.config import settings
from agentic_rag.core.logging import configure_logging
from agentic_rag.middleware.request_logging import MAX_REQUEST_ID_LENGTH, request_logging_middleware

pytestmark = pytest.mark.no_db


def _request_completed_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for record in caplog.records:
        message = record.msg
        if isinstance(message, dict) and message.get("event") == "request_completed":
            records.append(cast(dict[str, Any], message))

    return records


def _request_failed_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for record in caplog.records:
        message = record.msg
        if isinstance(message, dict) and message.get("event") == "request_failed":
            records.append(cast(dict[str, Any], message))

    return records


async def test_successful_request_is_logged_and_returns_request_id(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = await client.get("/health")

    request_id = response.headers["X-Request-ID"]
    request_record = _request_completed_records(caplog)[-1]

    UUID(request_id)
    assert request_record["request_id"] == request_id
    assert request_record["http_method"] == "GET"
    assert request_record["http_path"] == "/health"
    assert request_record["status_code"] == 200


async def test_valid_request_id_header_is_used(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "client-request_123"})

    request_record = _request_completed_records(caplog)[-1]

    assert response.headers["X-Request-ID"] == "client-request_123"
    assert request_record["request_id"] == "client-request_123"


async def test_invalid_request_id_header_is_replaced(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_request_id = "x" * (MAX_REQUEST_ID_LENGTH + 1)

    response = await client.get("/health", headers={"X-Request-ID": invalid_request_id})

    request_id = response.headers["X-Request-ID"]
    request_record = _request_completed_records(caplog)[-1]

    UUID(request_id)
    assert request_id != invalid_request_id
    assert request_record["request_id"] == request_id
    assert invalid_request_id not in str(request_record)


async def test_404_request_is_logged_with_status(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = await client.get("/missing")

    request_record = _request_completed_records(caplog)[-1]

    assert response.status_code == 404
    assert request_record["status_code"] == 404


async def test_422_request_is_logged_without_body_leak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Payload(BaseModel):
        value: str = Field(min_length=1)

    test_app = FastAPI()
    test_app.middleware("http")(request_logging_middleware)

    @test_app.post("/validate")
    async def validate(_: Payload) -> dict[str, str]:
        return {"status": "ok"}

    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as client:
        response = await client.post("/validate", json={"value": ""})

    request_record = _request_completed_records(caplog)[-1]

    assert response.status_code == 422
    assert request_record["status_code"] == 422
    assert request_record["http_method"] == "POST"
    assert request_record["http_path"] == "/validate"


async def test_unhandled_exception_is_logged_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_app = FastAPI()
    test_app.middleware("http")(request_logging_middleware)

    @test_app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as client:
        with pytest.raises(RuntimeError, match="boom"):
            await client.get("/boom", headers={"X-Request-ID": "failed-request"})

    failed_record = _request_failed_records(caplog)[-1]

    assert failed_record["request_id"] == "failed-request"
    assert failed_record["http_method"] == "GET"
    assert failed_record["http_path"] == "/boom"
    assert failed_record["status_code"] == 500
    assert "exception" in failed_record
    assert "RuntimeError: boom" in str(failed_record["exception"])


async def test_sensitive_headers_are_not_logged(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = await client.get(
        "/health",
        headers={
            "Authorization": "Bearer secret-token",
            "Cookie": "session=secret-cookie",
            "X-Request-ID": "safe-request",
        },
    )

    request_record = _request_completed_records(caplog)[-1]

    assert response.status_code == 200
    assert request_record["request_id"] == "safe-request"
    assert "secret-token" not in str(request_record)
    assert "secret-cookie" not in str(request_record)
    assert "Authorization" not in request_record
    assert "Cookie" not in request_record


async def test_request_context_does_not_leak_between_requests(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await client.get("/health", headers={"X-Request-ID": "request-one"})
    await client.get("/health", headers={"X-Request-ID": "request-two"})

    first_record, second_record = _request_completed_records(caplog)[-2:]

    assert first_record["request_id"] == "request-one"
    assert second_record["request_id"] == "request-two"
    assert "request-one" not in str(second_record)


async def test_json_request_log_is_valid_json(
    client: AsyncClient,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "log_format", "json")
    configure_logging(settings)
    capfd.readouterr()

    try:
        response = await client.get("/health", headers={"X-Request-ID": "json-request-1"})

        records = [json.loads(line) for line in capfd.readouterr().out.splitlines()]
        request_record = next(
            record for record in records if record["event"] == "request_completed"
        )

        assert response.status_code == 200
        assert request_record["request_id"] == "json-request-1"
        assert request_record["http_method"] == "GET"
        assert request_record["http_path"] == "/health"
        assert request_record["status_code"] == 200
        assert request_record["service"] == settings.app_name
        assert request_record["environment"] == settings.environment
        assert isinstance(request_record["duration_ms"], float)
    finally:
        monkeypatch.setattr(settings, "log_format", "console")
        configure_logging(settings)
