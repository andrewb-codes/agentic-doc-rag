from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.repositories.user import UserRepository
from agentic_rag.services.user import UserService


def create_user_service(*, session: AsyncSession) -> UserService:
    return UserService(session=session, repository=UserRepository(session=session))


async def test_user_service_creates_telegram_user(
    session: AsyncSession,
) -> None:
    service = create_user_service(session=session)

    user = await service.get_or_create_telegram_user(
        telegram_user_id=123,
        telegram_username="andrew",
    )

    assert user.id == 1
    assert user.telegram_user_id == 123
    assert user.username == "andrew"
    assert user.is_active is True


async def test_user_service_returns_existing_telegram_user_and_updates_username(
    session: AsyncSession,
) -> None:
    service = create_user_service(session=session)
    user = await service.get_or_create_telegram_user(
        telegram_user_id=123,
        telegram_username="old",
    )

    same_user = await service.get_or_create_telegram_user(
        telegram_user_id=123,
        telegram_username="new",
    )

    assert same_user.id == user.id
    assert same_user.telegram_user_id == 123
    assert same_user.username == "new"
