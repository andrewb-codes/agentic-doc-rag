from datetime import datetime

from pydantic import BaseModel, Field

from agentic_rag.models import AnswerStatus, DocumentStatus
from agentic_rag.schemas.qa_history import VerificationResultResponse


class DocumentResponse(BaseModel):
    id: int
    owner_id: int
    filename: str
    status: DocumentStatus
    page_count: int | None
    chunk_count: int | None
    created_at: datetime


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class DocumentChunkResponse(BaseModel):
    id: int
    document_id: int
    page: int
    chunk_index: int
    text: str
    source: str


class DocumentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class DocumentAskResponse(BaseModel):
    answer: str
    answer_status: AnswerStatus
    chunks: list[DocumentChunkResponse]
    verification_result: VerificationResultResponse
