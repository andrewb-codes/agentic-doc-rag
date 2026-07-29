from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from agentic_rag.models import DocumentChunk, VerificationVerdict
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.verification import (
    AnswerVerificationService,
    parse_verification_verdict,
)

pytestmark = pytest.mark.no_db


def test_parse_verification_verdict_returns_supported() -> None:
    assert parse_verification_verdict(" supported ") == VerificationVerdict.SUPPORTED


def test_parse_verification_verdict_returns_unsupported() -> None:
    assert parse_verification_verdict("unsupported") == VerificationVerdict.UNSUPPORTED


def test_parse_verification_verdict_treats_unknown_value_as_unsupported() -> None:
    assert parse_verification_verdict("maybe") == VerificationVerdict.UNSUPPORTED


async def test_answer_verification_service_verifies_answer() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    verify_answer = AsyncMock(return_value="supported")

    service = AnswerVerificationService(
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
    )

    verdict = await service.verify_answer(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )

    verify_answer.assert_awaited_once_with(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )
    assert verdict == VerificationVerdict.SUPPORTED


async def test_answer_verification_service_verifies_answer_without_chunks() -> None:
    verify_answer = AsyncMock(return_value="unsupported")

    service = AnswerVerificationService(
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
    )

    verdict = await service.verify_answer(
        question="When did Atlas start?",
        answer="Answer not found in documents.",
        chunks=[],
    )

    verify_answer.assert_awaited_once_with(
        question="When did Atlas start?",
        answer="Answer not found in documents.",
        chunks=[],
    )
    assert verdict == VerificationVerdict.UNSUPPORTED
