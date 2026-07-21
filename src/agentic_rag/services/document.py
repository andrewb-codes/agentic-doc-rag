from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.core.exceptions import DocumentNotFoundError
from agentic_rag.models import Document, DocumentStatus
from agentic_rag.repositories.document import DocumentRepository
from agentic_rag.services.pdf import PdfExtractionError, PdfExtractor


class DocumentService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = DocumentRepository(session)

    async def get_user_document(self, *, document_id: int, owner_id: int) -> Document:
        document = await self.repository.get_owned_by_id(
            document_id=document_id,
            owner_id=owner_id,
        )

        if document is None:
            raise DocumentNotFoundError()

        return document

    async def list_user_documents(self, *, owner_id: int) -> list[Document]:
        return await self.repository.list_by_owner(owner_id=owner_id)

    async def create_document_metadata(self, *, owner_id: int, filename: str) -> Document:
        document = await self.repository.create_document_metadata(
            owner_id=owner_id, filename=filename
        )
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def process_uploaded_pdf(
        self,
        *,
        owner_id: int,
        filename: str,
        path: Path,
    ) -> Document:
        document = await self.repository.create_document_metadata(
            owner_id=owner_id,
            filename=filename,
            status=DocumentStatus.PROCESSING,
        )

        try:
            extracted = PdfExtractor().extract(path=path)
        except PdfExtractionError:
            await self.repository.update_processing_result(
                document=document,
                status=DocumentStatus.FAILED,
            )
            await self.session.commit()
            await self.session.refresh(document)
            raise

        await self.repository.update_processing_result(
            document=document,
            status=DocumentStatus.PROCESSED,
            page_count=extracted.page_count,
            chunk_count=0,
        )
        await self.session.commit()
        await self.session.refresh(document)
        return document
