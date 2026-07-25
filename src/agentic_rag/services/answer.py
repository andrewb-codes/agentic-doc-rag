from dataclasses import dataclass

from agentic_rag.models import DocumentChunk
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.retrieval import RetrievalService


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    chunks: list[DocumentChunk]


class AnswerService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        chat_service: OpenAIChatService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.chat_service = chat_service

    async def answer_user_question(
        self,
        *,
        question: str,
        owner_id: int,
        limit: int,
    ) -> AnswerResult:
        chunks = await self.retrieval_service.search_user_chunks(
            query=question,
            owner_id=owner_id,
            limit=limit,
        )
        answer = await self.chat_service.answer_question(question=question, chunks=chunks)

        return AnswerResult(answer=answer, chunks=chunks)

    async def answer_document_question(
        self,
        *,
        question: str,
        owner_id: int,
        document_id: int,
        limit: int,
    ) -> AnswerResult:
        chunks = await self.retrieval_service.search_document_chunks(
            query=question,
            owner_id=owner_id,
            document_id=document_id,
            limit=limit,
        )
        answer = await self.chat_service.answer_question(question=question, chunks=chunks)

        return AnswerResult(answer=answer, chunks=chunks)
