from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, UploadFile, status

from agentic_rag.api.deps import (
    get_answer_service,
    get_current_telegram_user,
    get_document_metadata_service,
    get_document_processing_service,
    get_retrieval_service,
)
from agentic_rag.api.presenters.documents import (
    build_document_ask_response,
    build_document_chunk_response,
    build_document_response,
)
from agentic_rag.api.upload import save_upload_to_temp_file, validate_pdf_upload
from agentic_rag.core.exceptions import PdfProcessingError
from agentic_rag.models import User
from agentic_rag.schemas.document import (
    DocumentAskRequest,
    DocumentAskResponse,
    DocumentChunkResponse,
    DocumentCreateRequest,
    DocumentResponse,
    DocumentSearchRequest,
)
from agentic_rag.services.answer import AnswerService
from agentic_rag.services.document import DocumentMetadataService, DocumentProcessingService
from agentic_rag.services.pdf import PdfExtractionError
from agentic_rag.services.retrieval import RetrievalService

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
) -> list[DocumentResponse]:
    documents = await service.list_user_documents(owner_id=current_user.id)
    return [build_document_response(document) for document in documents]


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document_metadata(
    request: DocumentCreateRequest,
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
) -> DocumentResponse:
    document = await service.create_document_metadata(
        owner_id=current_user.id, filename=request.filename
    )

    return build_document_response(document)


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentProcessingService, Depends(get_document_processing_service)],
) -> DocumentResponse:
    validate_pdf_upload(file)
    temp_path = await save_upload_to_temp_file(file)

    try:
        document = await service.process_uploaded_pdf(
            owner_id=current_user.id,
            filename=file.filename or temp_path.name,
            path=temp_path,
        )
    except PdfExtractionError as exc:
        raise PdfProcessingError() from exc
    finally:
        temp_path.unlink(missing_ok=True)
        await file.close()

    return build_document_response(document)


@router.post("/search", response_model=list[DocumentChunkResponse])
async def search_documents(
    request: DocumentSearchRequest,
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> list[DocumentChunkResponse]:
    chunks = await service.search_user_chunks(
        query=request.query,
        owner_id=current_user.id,
        limit=request.limit,
    )

    return [build_document_chunk_response(chunk) for chunk in chunks]


@router.post("/ask", response_model=DocumentAskResponse)
async def ask_documents(
    request: DocumentAskRequest,
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[AnswerService, Depends(get_answer_service)],
) -> DocumentAskResponse:
    result = await service.answer_user_question(
        question=request.question,
        owner_id=current_user.id,
        limit=request.limit,
    )

    return build_document_ask_response(result)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: Annotated[int, Path(gt=0)],
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
) -> DocumentResponse:
    document = await service.get_user_document(document_id=document_id, owner_id=current_user.id)
    return build_document_response(document)


@router.post("/{document_id}/search", response_model=list[DocumentChunkResponse])
async def search_document(
    document_id: Annotated[int, Path(gt=0)],
    request: DocumentSearchRequest,
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    metadata_service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> list[DocumentChunkResponse]:
    await metadata_service.get_user_document(
        document_id=document_id,
        owner_id=current_user.id,
    )
    chunks = await retrieval_service.search_document_chunks(
        query=request.query,
        owner_id=current_user.id,
        document_id=document_id,
        limit=request.limit,
    )

    return [build_document_chunk_response(chunk) for chunk in chunks]


@router.post("/{document_id}/ask", response_model=DocumentAskResponse)
async def ask_document(
    document_id: Annotated[int, Path(gt=0)],
    request: DocumentAskRequest,
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    metadata_service: Annotated[DocumentMetadataService, Depends(get_document_metadata_service)],
    answer_service: Annotated[AnswerService, Depends(get_answer_service)],
) -> DocumentAskResponse:
    await metadata_service.get_user_document(
        document_id=document_id,
        owner_id=current_user.id,
    )
    result = await answer_service.answer_document_question(
        question=request.question,
        owner_id=current_user.id,
        document_id=document_id,
        limit=request.limit,
    )

    return build_document_ask_response(result)
