from agentic_rag.models import QAHistory
from agentic_rag.schemas.qa_history import (
    QAHistoryResponse,
    UnsupportedClaimResponse,
    VerificationResultResponse,
)


def build_verification_result_response(qa_history: QAHistory) -> VerificationResultResponse:
    return VerificationResultResponse(
        verdict=qa_history.verification_verdict,
        unsupported_claims=[
            UnsupportedClaimResponse.model_validate(unsupported_claim)
            for unsupported_claim in qa_history.unsupported_claims
        ],
        missing_information=qa_history.missing_information,
        confidence=qa_history.verification_confidence,
    )


def build_qa_history_response(qa_history: QAHistory) -> QAHistoryResponse:
    return QAHistoryResponse(
        id=qa_history.id,
        user_id=qa_history.user_id,
        document_id=qa_history.document_id,
        question=qa_history.question,
        answer=qa_history.answer,
        verification_result=build_verification_result_response(qa_history),
        created_at=qa_history.created_at,
    )
