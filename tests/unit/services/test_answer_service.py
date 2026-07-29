from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import DocumentChunk, VerificationVerdict
from agentic_rag.repositories.qa_history import QAHistoryRepository
from agentic_rag.services.answer import (
    NO_DOCUMENT_ANSWER_FOUND,
    NO_USER_ANSWER_FOUND,
    AnswerService,
)
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.retrieval import RetrievalService
from agentic_rag.services.verification import AnswerVerificationService

pytestmark = pytest.mark.no_db


async def test_answer_service_answers_question_from_user_chunks() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    search_user_chunks = AsyncMock(return_value=[chunk])
    answer_question = AsyncMock(return_value="Atlas started on March 14, 2025.")
    verify_answer = AsyncMock(return_value=VerificationVerdict.SUPPORTED)
    create_history = AsyncMock()
    commit = AsyncMock()

    service = AnswerService(
        session=cast(AsyncSession, cast(object, SimpleNamespace(commit=commit))),
        retrieval_service=cast(
            RetrievalService,
            cast(object, SimpleNamespace(search_user_chunks=search_user_chunks)),
        ),
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(answer_question=answer_question)),
        ),
        verification_service=cast(
            AnswerVerificationService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
        qa_history_repository=cast(
            QAHistoryRepository,
            cast(object, SimpleNamespace(create=create_history)),
        ),
    )

    result = await service.answer_user_question(
        question="When did Atlas start?",
        owner_id=1,
        limit=5,
    )

    search_user_chunks.assert_awaited_once_with(
        query="When did Atlas start?",
        owner_id=1,
        limit=5,
    )
    answer_question.assert_awaited_once_with(
        question="When did Atlas start?",
        chunks=[chunk],
    )
    verify_answer.assert_awaited_once_with(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )
    create_history.assert_awaited_once_with(
        user_id=1,
        document_id=None,
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        verification_verdict=VerificationVerdict.SUPPORTED,
    )
    commit.assert_awaited_once_with()
    assert result.answer == "Atlas started on March 14, 2025."
    assert result.chunks == [chunk]
    assert result.verification_verdict == VerificationVerdict.SUPPORTED


async def test_answer_service_answers_question_from_document_chunks() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    search_document_chunks = AsyncMock(return_value=[chunk])
    answer_question = AsyncMock(return_value="Atlas started on March 14, 2025.")
    verify_answer = AsyncMock(return_value=VerificationVerdict.SUPPORTED)
    create_history = AsyncMock()
    commit = AsyncMock()

    service = AnswerService(
        session=cast(AsyncSession, cast(object, SimpleNamespace(commit=commit))),
        retrieval_service=cast(
            RetrievalService,
            cast(object, SimpleNamespace(search_document_chunks=search_document_chunks)),
        ),
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(answer_question=answer_question)),
        ),
        verification_service=cast(
            AnswerVerificationService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
        qa_history_repository=cast(
            QAHistoryRepository,
            cast(object, SimpleNamespace(create=create_history)),
        ),
    )

    result = await service.answer_document_question(
        question="When did Atlas start?",
        owner_id=1,
        document_id=20,
        limit=5,
    )

    search_document_chunks.assert_awaited_once_with(
        query="When did Atlas start?",
        owner_id=1,
        document_id=20,
        limit=5,
    )
    answer_question.assert_awaited_once_with(
        question="When did Atlas start?",
        chunks=[chunk],
    )
    verify_answer.assert_awaited_once_with(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )
    create_history.assert_awaited_once_with(
        user_id=1,
        document_id=20,
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        verification_verdict=VerificationVerdict.SUPPORTED,
    )
    commit.assert_awaited_once_with()
    assert result.answer == "Atlas started on March 14, 2025."
    assert result.chunks == [chunk]
    assert result.verification_verdict == VerificationVerdict.SUPPORTED


async def test_answer_service_returns_unsupported_without_user_chunks() -> None:
    search_user_chunks = AsyncMock(return_value=[])
    answer_question = AsyncMock()
    verify_answer = AsyncMock()
    create_history = AsyncMock()
    commit = AsyncMock()

    service = AnswerService(
        session=cast(AsyncSession, cast(object, SimpleNamespace(commit=commit))),
        retrieval_service=cast(
            RetrievalService,
            cast(object, SimpleNamespace(search_user_chunks=search_user_chunks)),
        ),
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(answer_question=answer_question)),
        ),
        verification_service=cast(
            AnswerVerificationService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
        qa_history_repository=cast(
            QAHistoryRepository,
            cast(object, SimpleNamespace(create=create_history)),
        ),
    )

    result = await service.answer_user_question(
        question="When did Atlas start?",
        owner_id=1,
        limit=5,
    )

    search_user_chunks.assert_awaited_once_with(
        query="When did Atlas start?",
        owner_id=1,
        limit=5,
    )
    answer_question.assert_not_awaited()
    verify_answer.assert_not_awaited()
    create_history.assert_awaited_once_with(
        user_id=1,
        document_id=None,
        question="When did Atlas start?",
        answer=NO_USER_ANSWER_FOUND,
        verification_verdict=VerificationVerdict.UNSUPPORTED,
    )
    commit.assert_awaited_once_with()
    assert result.answer == NO_USER_ANSWER_FOUND
    assert result.chunks == []
    assert result.verification_verdict == VerificationVerdict.UNSUPPORTED


async def test_answer_service_returns_unsupported_without_document_chunks() -> None:
    search_document_chunks = AsyncMock(return_value=[])
    answer_question = AsyncMock()
    verify_answer = AsyncMock()
    create_history = AsyncMock()
    commit = AsyncMock()

    service = AnswerService(
        session=cast(AsyncSession, cast(object, SimpleNamespace(commit=commit))),
        retrieval_service=cast(
            RetrievalService,
            cast(object, SimpleNamespace(search_document_chunks=search_document_chunks)),
        ),
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(answer_question=answer_question)),
        ),
        verification_service=cast(
            AnswerVerificationService,
            cast(object, SimpleNamespace(verify_answer=verify_answer)),
        ),
        qa_history_repository=cast(
            QAHistoryRepository,
            cast(object, SimpleNamespace(create=create_history)),
        ),
    )

    result = await service.answer_document_question(
        question="When did Atlas start?",
        owner_id=1,
        document_id=20,
        limit=5,
    )

    search_document_chunks.assert_awaited_once_with(
        query="When did Atlas start?",
        owner_id=1,
        document_id=20,
        limit=5,
    )
    answer_question.assert_not_awaited()
    verify_answer.assert_not_awaited()
    create_history.assert_awaited_once_with(
        user_id=1,
        document_id=20,
        question="When did Atlas start?",
        answer=NO_DOCUMENT_ANSWER_FOUND,
        verification_verdict=VerificationVerdict.UNSUPPORTED,
    )
    commit.assert_awaited_once_with()
    assert result.answer == NO_DOCUMENT_ANSWER_FOUND
    assert result.chunks == []
    assert result.verification_verdict == VerificationVerdict.UNSUPPORTED
