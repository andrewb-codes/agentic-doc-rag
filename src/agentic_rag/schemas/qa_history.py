from datetime import datetime

from pydantic import BaseModel

from agentic_rag.models import VerificationVerdict


class UnsupportedClaimResponse(BaseModel):
    claim: str
    reason: str


class VerificationResultResponse(BaseModel):
    verdict: VerificationVerdict
    unsupported_claims: list[UnsupportedClaimResponse]
    missing_information: list[str]
    confidence: float | None


class QAHistoryResponse(BaseModel):
    id: int
    user_id: int
    document_id: int | None
    question: str
    answer: str
    verification_result: VerificationResultResponse
    created_at: datetime
