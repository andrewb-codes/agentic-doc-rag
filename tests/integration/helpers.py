from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import Document, DocumentStatus, User


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
