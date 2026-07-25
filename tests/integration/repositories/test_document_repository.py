from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import DocumentStatus
from agentic_rag.repositories.document import DocumentRepository
from tests.integration.helpers import create_user


async def test_document_repository_get_owned_by_id_returns_only_owner_document(
    session: AsyncSession,
) -> None:
    first_user = await create_user(session, telegram_user_id=111, username="first")
    second_user = await create_user(session, telegram_user_id=222, username="second")

    repository = DocumentRepository(session=session)
    document = await repository.create_document_metadata(
        owner_id=first_user.id,
        filename="manual.pdf",
    )

    owner_document = await repository.get_owned_by_id(
        document_id=document.id,
        owner_id=first_user.id,
    )
    foreign_document = await repository.get_owned_by_id(
        document_id=document.id,
        owner_id=second_user.id,
    )

    assert owner_document == document
    assert foreign_document is None


async def test_document_repository_lists_documents_by_owner_descending(
    session: AsyncSession,
) -> None:
    first_user = await create_user(session, telegram_user_id=111, username="first")
    second_user = await create_user(session, telegram_user_id=222, username="second")

    repository = DocumentRepository(session=session)
    first_document = await repository.create_document_metadata(
        owner_id=first_user.id,
        filename="first.pdf",
    )
    second_document = await repository.create_document_metadata(
        owner_id=first_user.id,
        filename="second.pdf",
    )
    await repository.create_document_metadata(
        owner_id=second_user.id,
        filename="foreign.pdf",
    )

    documents = await repository.list_by_owner(owner_id=first_user.id)

    assert documents == [second_document, first_document]


async def test_document_repository_updates_processing_result(
    session: AsyncSession,
) -> None:
    user = await create_user(session)

    repository = DocumentRepository(session=session)
    document = await repository.create_document_metadata(
        owner_id=user.id,
        filename="manual.pdf",
        status=DocumentStatus.PROCESSING,
    )

    updated_document = await repository.update_processing_result(
        document=document,
        status=DocumentStatus.PROCESSED,
        page_count=2,
        chunk_count=5,
    )

    assert updated_document == document
    assert document.status == DocumentStatus.PROCESSED
    assert document.page_count == 2
    assert document.chunk_count == 5
