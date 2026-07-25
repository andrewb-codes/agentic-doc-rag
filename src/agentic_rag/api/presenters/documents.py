from agentic_rag.models import Document, DocumentChunk
from agentic_rag.schemas.document import DocumentChunkResponse, DocumentResponse


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


def build_document_chunk_response(chunk: DocumentChunk) -> DocumentChunkResponse:
    return DocumentChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        page=chunk.page,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        source=chunk.source,
    )
