from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, UploadFile, status

from agentic_rag.api.deps import (
    get_current_telegram_user,
    get_document_service,
)
from agentic_rag.api.presenters.documents import build_document_response
from agentic_rag.api.upload import save_upload_to_temp_file, validate_pdf_upload
from agentic_rag.core.exceptions import PdfProcessingError
from agentic_rag.models import User
from agentic_rag.schemas.document import DocumentCreateRequest, DocumentResponse
from agentic_rag.services.document import DocumentService
from agentic_rag.services.pdf import PdfExtractionError

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: Annotated[int, Path(gt=0)],
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    document = await service.get_user_document(document_id=document_id, owner_id=current_user.id)
    return build_document_response(document)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> list[DocumentResponse]:
    documents = await service.list_user_documents(owner_id=current_user.id)
    return [build_document_response(document) for document in documents]


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document_metadata(
    request: DocumentCreateRequest,
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    document = await service.create_document_metadata(
        owner_id=current_user.id, filename=request.filename
    )
    return build_document_response(document)


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[DocumentService, Depends(get_document_service)],
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
