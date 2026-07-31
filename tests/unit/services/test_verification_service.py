from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from agentic_rag.models import DocumentChunk, VerificationVerdict
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.verification import (
    AnswerVerificationService,
    UnsupportedClaim,
    parse_verification_result,
)

pytestmark = pytest.mark.no_db


def test_parse_verification_result_treats_non_json_as_unsupported() -> None:
    result = parse_verification_result(" supported ")

    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert result.unsupported_claims == []
    assert result.confidence is None


def test_parse_verification_result_returns_supported_from_json() -> None:
    result = parse_verification_result(
        """
        {
            "verdict": "supported",
            "unsupported_claims": [],
            "missing_information": [],
            "confidence": 0.95
        }
        """
    )

    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.unsupported_claims == []
    assert result.missing_information == []
    assert result.confidence == 0.95


def test_parse_verification_result_returns_unsupported_from_json() -> None:
    result = parse_verification_result(
        """
        {
            "verdict": "unsupported",
            "unsupported_claims": [
                {
                    "claim": "answer says March",
                    "reason": "context says April"
                }
            ],
            "missing_information": [],
            "confidence": 0.7
        }
        """
    )

    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert result.unsupported_claims == [
        UnsupportedClaim(claim="answer says March", reason="context says April")
    ]
    assert result.missing_information == []
    assert result.confidence == 0.7


@pytest.mark.parametrize(
    "value",
    [
        '{"verdict":"not_verified","unsupported_claims":[],"missing_information":[],"confidence":0.5}',
        '{"verdict":"supported","unsupported_claims":[],"missing_information":[],"confidence":2}',
        '{"verdict":"supported"}',
        '{"verdict":"unsupported","unsupported_claims":["old shape"],"missing_information":[],"confidence":0.5}',
        "[]",
    ],
)
def test_parse_verification_result_treats_invalid_json_shape_as_unsupported(
    value: str,
) -> None:
    result = parse_verification_result(value)

    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert result.unsupported_claims == []
    assert result.confidence is None


async def test_answer_verification_service_verifies_answer() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    verify_answer = AsyncMock(
        return_value="""
        {
            "verdict": "supported",
            "unsupported_claims": [],
            "missing_information": [],
            "confidence": 0.9
        }
        """
    )

    service = AnswerVerificationService(
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
    )

    result = await service.verify_answer(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )

    verify_answer.assert_awaited_once_with(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.confidence == 0.9


async def test_answer_verification_service_verifies_answer_without_chunks() -> None:
    verify_answer = AsyncMock(
        return_value="""
        {
            "verdict": "unsupported",
            "unsupported_claims": [],
            "missing_information": ["No relevant context."],
            "confidence": null
        }
        """
    )

    service = AnswerVerificationService(
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
    )

    result = await service.verify_answer(
        question="When did Atlas start?",
        answer="Answer not found in documents.",
        chunks=[],
    )

    verify_answer.assert_awaited_once_with(
        question="When did Atlas start?",
        answer="Answer not found in documents.",
        chunks=[],
    )
    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert result.missing_information == ["No relevant context."]
    assert result.confidence is None
