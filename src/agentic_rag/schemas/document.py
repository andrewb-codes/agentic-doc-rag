from datetime import datetime

from pydantic import BaseModel, Field

from agentic_rag.models import DocumentStatus


class DocumentCreateRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)


class DocumentResponse(BaseModel):
    id: int
    owner_id: int
    filename: str
    status: DocumentStatus
    page_count: int | None
    chunk_count: int | None
    created_at: datetime
