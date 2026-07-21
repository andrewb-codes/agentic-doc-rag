from agentic_rag.models import Document
from agentic_rag.schemas.document import DocumentResponse


def build_document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        owner_id=document.owner_id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )
