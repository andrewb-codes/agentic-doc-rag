from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import VerificationVerdict
from agentic_rag.repositories.qa_history import QAHistoryRepository
from tests.integration.helpers import create_document, create_user


async def test_qa_history_repository_creates_history_item(session: AsyncSession) -> None:
    user = await create_user(session)
    document = await create_document(session, owner_id=user.id)

    repository = QAHistoryRepository(session=session)

    item = await repository.create(
        user_id=user.id,
        document_id=document.id,
        question="Question?",
        answer="Answer.",
        verification_verdict=VerificationVerdict.NOT_VERIFIED,
    )

    assert item.id == 1
    assert item.user_id == user.id
    assert item.document_id == document.id
    assert item.question == "Question?"
    assert item.answer == "Answer."
    assert item.verification_verdict == VerificationVerdict.NOT_VERIFIED
    assert item.created_at is not None


async def test_qa_history_repository_creates_history_item_without_document(
    session: AsyncSession,
) -> None:
    user = await create_user(session)

    repository = QAHistoryRepository(session=session)

    item = await repository.create(
        user_id=user.id,
        document_id=None,
        question="Question?",
        answer="Answer.",
    )

    assert item.id == 1
    assert item.user_id == user.id
    assert item.document_id is None
    assert item.question == "Question?"
    assert item.answer == "Answer."
    assert item.verification_verdict == VerificationVerdict.NOT_VERIFIED
    assert item.created_at is not None
