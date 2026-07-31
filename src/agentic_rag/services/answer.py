from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import AnswerStatus, DocumentChunk, VerificationVerdict
from agentic_rag.repositories.qa_history import QAHistoryRepository
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.retrieval import RetrievalService
from agentic_rag.services.verification import AnswerVerificationService, VerificationResult

NO_USER_ANSWER_FOUND = "Ответ не найден в документах."
NO_DOCUMENT_ANSWER_FOUND = "Ответ не найден в документе."


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    answer_status: AnswerStatus
    chunks: list[DocumentChunk]
    verification_result: VerificationResult


class AnswerService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        retrieval_service: RetrievalService,
        chat_service: OpenAIChatService,
        verification_service: AnswerVerificationService,
        qa_history_repository: QAHistoryRepository,
    ) -> None:
        self.session = session
        self.retrieval_service = retrieval_service
        self.chat_service = chat_service
        self.verification_service = verification_service
        self.qa_history_repository = qa_history_repository

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

        answer, answer_status, verification_result = await self._build_answer(
            question=question,
            chunks=chunks,
            no_answer_text=NO_USER_ANSWER_FOUND,
        )

        await self.qa_history_repository.create(
            user_id=owner_id,
            document_id=None,
            question=question,
            answer=answer,
            verification_result=verification_result,
        )

        await self.session.commit()

        return AnswerResult(
            answer=answer,
            answer_status=answer_status,
            chunks=chunks,
            verification_result=verification_result,
        )

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

        answer, answer_status, verification_result = await self._build_answer(
            question=question,
            chunks=chunks,
            no_answer_text=NO_DOCUMENT_ANSWER_FOUND,
        )

        await self.qa_history_repository.create(
            user_id=owner_id,
            document_id=document_id,
            question=question,
            answer=answer,
            verification_result=verification_result,
        )

        await self.session.commit()

        return AnswerResult(
            answer=answer,
            answer_status=answer_status,
            chunks=chunks,
            verification_result=verification_result,
        )

    async def _build_answer(
        self,
        *,
        question: str,
        chunks: list[DocumentChunk],
        no_answer_text: str,
    ) -> tuple[str, AnswerStatus, VerificationResult]:
        if not chunks:
            return (
                no_answer_text,
                AnswerStatus.NOT_FOUND,
                VerificationResult(
                    verdict=VerificationVerdict.UNSUPPORTED,
                    unsupported_claims=[],
                    missing_information=["No relevant document chunks found."],
                    confidence=None,
                ),
            )

        answer = await self.chat_service.answer_question(question=question, chunks=chunks)
        answer_status = (
            AnswerStatus.NOT_FOUND if is_no_answer_text(answer) else AnswerStatus.ANSWERED
        )
        verification_result = await self.verification_service.verify_answer(
            question=question,
            answer=answer,
            chunks=chunks,
        )

        return answer, answer_status, verification_result


def is_no_answer_text(answer: str) -> bool:
    normalized = answer.strip().casefold()
    return (
        "не найден" in normalized
        or "не удалось найти" in normalized
        or "not found" in normalized
        or "does not contain the answer" in normalized
    )
