from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import Document, DocumentStatus, QAHistory, User, VerificationVerdict
from agentic_rag.repositories.qa_history import QAHistoryRepository
from agentic_rag.services.verification import VerificationResult


async def create_user(
    session: AsyncSession,
    *,
    telegram_user_id: int = 123,
    username: str | None = "andrew",
) -> User:
    user = User(telegram_user_id=telegram_user_id, username=username)
    session.add(user)
    await session.flush()
    return user


async def create_document(
    session: AsyncSession,
    *,
    owner_id: int,
    filename: str = "manual.pdf",
    status: DocumentStatus = DocumentStatus.PROCESSED,
) -> Document:
    document = Document(
        owner_id=owner_id,
        filename=filename,
        status=status,
    )
    session.add(document)
    await session.flush()
    return document


async def create_qa_history(
    session: AsyncSession,
    *,
    user_id: int,
    document_id: int | None = None,
    question: str = "Question?",
    answer: str = "Answer.",
    verification_result: VerificationResult | None = None,
) -> QAHistory:
    if verification_result is None:
        verification_result = VerificationResult(
            verdict=VerificationVerdict.UNSUPPORTED,
            unsupported_claims=[],
            missing_information=[],
            confidence=0.0,
        )

    return await QAHistoryRepository(session=session).create(
        user_id=user_id,
        document_id=document_id,
        question=question,
        answer=answer,
        verification_result=verification_result,
    )
