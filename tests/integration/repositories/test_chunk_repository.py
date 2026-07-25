from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import DocumentChunk
from agentic_rag.repositories.chunk import DocumentChunkRepository
from agentic_rag.services.chunk import TextChunk
from tests.integration.helpers import create_document, create_user


async def test_chunk_repository_creates_chunks(
    session: AsyncSession,
) -> None:
    user = await create_user(session)
    document = await create_document(session, owner_id=user.id)

    repository = DocumentChunkRepository(session=session)

    chunks = await repository.create_chunks(
        document_id=document.id,
        chunks_list=[
            TextChunk(
                page=1,
                chunk_index=0,
                text="first",
                source="manual.pdf",
            )
        ],
    )

    assert len(chunks) == 1
    assert chunks[0].document_id == document.id
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "first"
    assert chunks[0].source == "manual.pdf"


async def test_chunk_repository_get_by_ids_preserves_requested_order(
    session: AsyncSession,
) -> None:
    user = await create_user(session)
    document = await create_document(session, owner_id=user.id)

    first_chunk = DocumentChunk(
        document_id=document.id,
        page=1,
        chunk_index=0,
        text="first",
        source="manual.pdf",
    )
    second_chunk = DocumentChunk(
        document_id=document.id,
        page=1,
        chunk_index=1,
        text="second",
        source="manual.pdf",
    )
    session.add_all([first_chunk, second_chunk])
    await session.flush()

    repository = DocumentChunkRepository(session=session)

    chunks = await repository.get_by_ids(chunk_ids=[second_chunk.id, first_chunk.id])

    assert [chunk.id for chunk in chunks] == [second_chunk.id, first_chunk.id]


async def test_chunk_repository_get_by_ids_ignores_missing_ids(
    session: AsyncSession,
) -> None:
    user = await create_user(session)
    document = await create_document(session, owner_id=user.id)

    chunk = DocumentChunk(
        document_id=document.id,
        page=1,
        chunk_index=0,
        text="first",
        source="manual.pdf",
    )
    session.add(chunk)
    await session.flush()

    repository = DocumentChunkRepository(session=session)

    chunks = await repository.get_by_ids(chunk_ids=[999, chunk.id])

    assert chunks == [chunk]


async def test_chunk_repository_get_by_ids_returns_empty_list_for_empty_ids(
    session: AsyncSession,
) -> None:
    repository = DocumentChunkRepository(session=session)

    chunks = await repository.get_by_ids(chunk_ids=[])

    assert chunks == []
