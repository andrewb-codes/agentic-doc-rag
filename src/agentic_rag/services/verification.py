from agentic_rag.models import DocumentChunk, VerificationVerdict
from agentic_rag.services.llm import OpenAIChatService


class AnswerVerificationService:
    def __init__(self, *, chat_service: OpenAIChatService) -> None:
        self.chat_service = chat_service

    async def verify_answer(
        self,
        *,
        question: str,
        answer: str,
        chunks: list[DocumentChunk],
    ) -> VerificationVerdict:
        verdict = await self.chat_service.verify_answer(
            question=question,
            answer=answer,
            chunks=chunks,
        )

        return parse_verification_verdict(verdict)


def parse_verification_verdict(value: str) -> VerificationVerdict:
    normalized = value.strip().lower()

    if normalized == VerificationVerdict.SUPPORTED:
        return VerificationVerdict.SUPPORTED

    return VerificationVerdict.UNSUPPORTED
