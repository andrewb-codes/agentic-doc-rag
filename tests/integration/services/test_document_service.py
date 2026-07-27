from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import Document, DocumentChunk, DocumentStatus
from agentic_rag.repositories.chunk import DocumentChunkRepository
from agentic_rag.repositories.document import DocumentRepository
from agentic_rag.services.document import DocumentProcessingService
from agentic_rag.services.indexing import DocumentIndexingService
from agentic_rag.services.pdf import InvalidPdfError
from tests.helpers import create_user
from tests.helpers.pdf import create_pdf


class FakeIndexingService:
    def __init__(self) -> None:
        self.index_chunks = AsyncMock()


def create_document_processing_service(
    *,
    session: AsyncSession,
    indexing_service: FakeIndexingService,
) -> DocumentProcessingService:
    return DocumentProcessingService(
        session=session,
        document_repository=DocumentRepository(session=session),
        chunk_repository=DocumentChunkRepository(session=session),
        indexing_service=cast(DocumentIndexingService, cast(object, indexing_service)),
    )


async def test_document_processing_service_process_uploaded_pdf_persists_document_and_chunks(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    user = await create_user(session)

    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path)

    indexing_service = FakeIndexingService()
    service = create_document_processing_service(
        session=session,
        indexing_service=indexing_service,
    )

    document = await service.process_uploaded_pdf(
        owner_id=user.id,
        filename="manual.pdf",
        path=pdf_path,
    )

    chunks = list(
        await session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
    )

    assert document.status == DocumentStatus.PROCESSED
    assert document.page_count == 1
    assert document.chunk_count == 1
    assert len(chunks) == 1
    assert chunks[0].text == "PDF text"
    indexing_service.index_chunks.assert_awaited_once_with(chunks=chunks, owner_id=user.id)


async def test_document_processing_service_marks_document_failed_when_pdf_is_invalid(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    user = await create_user(session)

    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf")

    indexing_service = FakeIndexingService()
    service = create_document_processing_service(
        session=session,
        indexing_service=indexing_service,
    )

    with pytest.raises(InvalidPdfError):
        await service.process_uploaded_pdf(
            owner_id=user.id,
            filename="broken.pdf",
            path=pdf_path,
        )

    documents = list(await session.scalars(select(Document).where(Document.owner_id == user.id)))

    assert len(documents) == 1
    assert documents[0].status == DocumentStatus.FAILED
    assert documents[0].page_count is None
    assert documents[0].chunk_count is None
