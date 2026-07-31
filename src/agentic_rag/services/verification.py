from pydantic import BaseModel, Field, ValidationError

from agentic_rag.models import DocumentChunk, VerificationVerdict
from agentic_rag.services.llm import OpenAIChatService


class UnsupportedClaim(BaseModel):
    claim: str
    reason: str


class VerificationResult(BaseModel):
    verdict: VerificationVerdict
    unsupported_claims: list[UnsupportedClaim]
    missing_information: list[str]
    confidence: float | None = Field(default=None, ge=0, le=1)


class AnswerVerificationService:
    def __init__(self, *, chat_service: OpenAIChatService) -> None:
        self.chat_service = chat_service

    async def verify_answer(
        self,
        *,
        question: str,
        answer: str,
        chunks: list[DocumentChunk],
    ) -> VerificationResult:
        result = await self.chat_service.verify_answer(
            question=question,
            answer=answer,
            chunks=chunks,
        )

        return parse_verification_result(result)


def parse_verification_result(value: str) -> VerificationResult:
    try:
        result = VerificationResult.model_validate_json(value)
    except ValidationError:
        return VerificationResult(
            verdict=VerificationVerdict.UNSUPPORTED,
            unsupported_claims=[],
            missing_information=[],
            confidence=None,
        )
    return result
