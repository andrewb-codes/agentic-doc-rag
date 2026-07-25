from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import DocumentChunk, DocumentStatus
from agentic_rag.services.document import DocumentService
from agentic_rag.services.pdf import InvalidPdfError
from tests.helpers.pdf import create_pdf
from tests.integration.helpers import create_user


class FakeIndexingService:
    def __init__(self) -> None:
        self.index_chunks = AsyncMock()


async def test_document_service_process_uploaded_pdf_persists_document_and_chunks(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    user = await create_user(session)

    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path)

    indexing_service = FakeIndexingService()
    service = DocumentService(
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
    indexing_service.index_chunks.assert_awaited_once_with(chunks=chunks)


async def test_document_service_marks_document_failed_when_pdf_is_invalid(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    user = await create_user(session)

    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf")

    service = DocumentService(
        session=session,
        indexing_service=FakeIndexingService(),
    )

    with pytest.raises(InvalidPdfError):
        await service.process_uploaded_pdf(
            owner_id=user.id,
            filename="broken.pdf",
            path=pdf_path,
        )

    documents = await service.list_user_documents(owner_id=user.id)

    assert len(documents) == 1
    assert documents[0].status == DocumentStatus.FAILED
    assert documents[0].page_count is None
    assert documents[0].chunk_count is None
