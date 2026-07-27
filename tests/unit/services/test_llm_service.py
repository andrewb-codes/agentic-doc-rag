from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError

from agentic_rag.core.exceptions import LLMProviderTimeoutError
from agentic_rag.models import DocumentChunk
from agentic_rag.services.llm import (
    ChatCompletionsClient,
    OpenAIChatService,
    build_rag_messages,
    build_verification_messages,
)

pytestmark = pytest.mark.no_db


def test_build_rag_messages_includes_question_and_chunks() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )

    messages = build_rag_messages(question="When did Atlas start?", chunks=[chunk])

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "When did Atlas start?" in messages[1]["content"]
    assert "chunk_id=10" in messages[1]["content"]
    assert "document_id=20" in messages[1]["content"]
    assert "page=3" in messages[1]["content"]
    assert "Project Atlas started on March 14, 2025." in messages[1]["content"]


async def test_openai_chat_service_sends_rag_messages_and_returns_answer() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Atlas started on March 14, 2025."))
            ]
        )
    )
    chat_completions_client = SimpleNamespace(create=create)

    service = OpenAIChatService(
        api_key="test-key",
        model="gpt-4o-mini",
        max_tokens=1000,
        base_url="https://provider.example/v1",
        chat_completions_client=cast(ChatCompletionsClient, cast(object, chat_completions_client)),
    )

    answer = await service.answer_question(
        question="When did Atlas start?",
        chunks=[chunk],
    )

    create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=build_rag_messages(
            question="When did Atlas start?",
            chunks=[chunk],
        ),
        max_tokens=1000,
    )
    assert answer == "Atlas started on March 14, 2025."


async def test_openai_chat_service_converts_timeout_error() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    create = AsyncMock(
        side_effect=APITimeoutError(request=httpx.Request("POST", "https://provider.test"))
    )
    chat_completions_client = SimpleNamespace(create=create)

    service = OpenAIChatService(
        api_key="test-key",
        model="gpt-4o-mini",
        max_tokens=1000,
        base_url="https://provider.example/v1",
        chat_completions_client=cast(ChatCompletionsClient, cast(object, chat_completions_client)),
    )

    with pytest.raises(LLMProviderTimeoutError):
        await service.answer_question(
            question="When did Atlas start?",
            chunks=[chunk],
        )


def test_build_verification_messages_includes_question_answer_and_chunks() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )

    messages = build_verification_messages(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "When did Atlas start?" in messages[1]["content"]
    assert "Atlas started on March 14, 2025." in messages[1]["content"]
    assert "source=manual.pdf" in messages[1]["content"]
    assert "page=3" in messages[1]["content"]
    assert "Project Atlas started on March 14, 2025." in messages[1]["content"]


async def test_openai_chat_service_sends_verification_messages_and_returns_verdict() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="Project Atlas started on March 14, 2025.",
        source="manual.pdf",
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="supported"))]
        )
    )
    chat_completions_client = SimpleNamespace(create=create)

    service = OpenAIChatService(
        api_key="test-key",
        model="gpt-4o-mini",
        max_tokens=1000,
        base_url="https://provider.example/v1",
        chat_completions_client=cast(ChatCompletionsClient, cast(object, chat_completions_client)),
    )

    verdict = await service.verify_answer(
        question="When did Atlas start?",
        answer="Atlas started on March 14, 2025.",
        chunks=[chunk],
    )

    create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=build_verification_messages(
            question="When did Atlas start?",
            answer="Atlas started on March 14, 2025.",
            chunks=[chunk],
        ),
        max_tokens=16,
    )
    assert verdict == "supported"
