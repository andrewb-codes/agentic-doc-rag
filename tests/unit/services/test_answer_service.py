from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from agentic_rag.models import DocumentChunk
from agentic_rag.services.answer import AnswerService
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.retrieval import RetrievalService

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

    service = AnswerService(
        retrieval_service=cast(
            RetrievalService,
            cast(object, SimpleNamespace(search_user_chunks=search_user_chunks)),
        ),
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(answer_question=answer_question)),
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
    assert result.answer == "Atlas started on March 14, 2025."
    assert result.chunks == [chunk]


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

    service = AnswerService(
        retrieval_service=cast(
            RetrievalService,
            cast(object, SimpleNamespace(search_document_chunks=search_document_chunks)),
        ),
        chat_service=cast(
            OpenAIChatService,
            cast(object, SimpleNamespace(answer_question=answer_question)),
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
    assert result.answer == "Atlas started on March 14, 2025."
    assert result.chunks == [chunk]
