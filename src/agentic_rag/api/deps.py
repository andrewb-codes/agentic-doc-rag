from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import UnauthorizedError
from agentic_rag.db.session import get_db_session
from agentic_rag.models import User
from agentic_rag.services.document import DocumentService
from agentic_rag.services.user import UserService


def verify_internal_api_key(
    x_internal_api_key: Annotated[str | None, Header(alias="X-Internal-Api-Key")] = None,
) -> None:
    if x_internal_api_key != settings.internal_api_key:
        raise UnauthorizedError()


def get_session(session: Annotated[AsyncSession, Depends(get_db_session)]) -> AsyncSession:
    return session


def get_user_service(session: Annotated[AsyncSession, Depends(get_session)]) -> UserService:
    return UserService(session)


def get_document_service(session: Annotated[AsyncSession, Depends(get_session)]) -> DocumentService:
    return DocumentService(session)


async def get_current_telegram_user(
    service: Annotated[UserService, Depends(get_user_service)],
    _: Annotated[None, Depends(verify_internal_api_key)],
    telegram_user_id: Annotated[int | None, Header(alias="X-Telegram-User-Id")] = None,
    telegram_username: Annotated[str | None, Header(alias="X-Telegram-Username")] = None,
) -> User:
    if telegram_user_id is None or telegram_user_id <= 0:
        raise UnauthorizedError()

    return await service.get_or_create_telegram_user(
        telegram_user_id=telegram_user_id,
        telegram_username=telegram_username,
    )
