from agentic_rag.models import Document, DocumentChunk
from agentic_rag.schemas.document import (
    DocumentAskResponse,
    DocumentChunkResponse,
    DocumentResponse,
)
from agentic_rag.schemas.qa_history import UnsupportedClaimResponse, VerificationResultResponse
from agentic_rag.services.answer import AnswerResult
from agentic_rag.services.verification import VerificationResult


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


def build_verification_result_response(result: VerificationResult) -> VerificationResultResponse:
    return VerificationResultResponse(
        verdict=result.verdict,
        unsupported_claims=[
            UnsupportedClaimResponse.model_validate(unsupported_claim)
            for unsupported_claim in result.unsupported_claims
        ],
        missing_information=result.missing_information,
        confidence=result.confidence,
    )


def build_document_ask_response(result: AnswerResult) -> DocumentAskResponse:
    return DocumentAskResponse(
        answer=result.answer,
        answer_status=result.answer_status,
        chunks=[build_document_chunk_response(chunk) for chunk in result.chunks],
        verification_result=build_verification_result_response(result.verification_result),
    )
