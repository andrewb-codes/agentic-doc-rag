from datetime import datetime

from pydantic import BaseModel

from agentic_rag.models import VerificationVerdict


class QAHistoryResponse(BaseModel):
    id: int
    user_id: int
    document_id: int | None
    question: str
    answer: str
    verification_verdict: VerificationVerdict
    created_at: datetime
