from agentic_rag.models import QAHistory
from agentic_rag.schemas.qa_history import QAHistoryResponse


def build_qa_history_response(qa_history: QAHistory) -> QAHistoryResponse:
    return QAHistoryResponse(
        id=qa_history.id,
        user_id=qa_history.user_id,
        document_id=qa_history.document_id,
        question=qa_history.question,
        answer=qa_history.answer,
        verification_verdict=qa_history.verification_verdict,
        created_at=qa_history.created_at,
    )
