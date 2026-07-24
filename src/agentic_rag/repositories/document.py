from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import Document, DocumentStatus


class DocumentRepository:
    def __init__(self, *, session: AsyncSession) -> None:
        self.session = session

    async def get_owned_by_id(self, *, document_id: int, owner_id: int) -> Document | None:
        query = select(Document).where(
            Document.id == document_id,
            Document.owner_id == owner_id,
        )
        return cast(Document | None, await self.session.scalar(query))

    async def list_by_owner(self, *, owner_id: int) -> list[Document]:
        query = select(Document).where(Document.owner_id == owner_id).order_by(Document.id.desc())
        return list(await self.session.scalars(query))

    async def create_document_metadata(
        self,
        *,
        owner_id: int,
        filename: str,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> Document:
        document = Document(owner_id=owner_id, filename=filename, status=status)
        self.session.add(document)
        await self.session.flush()
        return document

    async def update_processing_result(
        self,
        *,
        document: Document,
        status: DocumentStatus,
        page_count: int | None = None,
        chunk_count: int | None = None,
    ) -> Document:
        document.status = status
        document.page_count = page_count
        document.chunk_count = chunk_count
        await self.session.flush()
        return document
