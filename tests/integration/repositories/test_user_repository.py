from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.repositories.user import UserRepository


async def test_user_repository_creates_and_gets_user_by_telegram_id(
    session: AsyncSession,
) -> None:
    repository = UserRepository(session=session)

    user = await repository.create(
        telegram_user_id=123,
        username="andrew",
    )

    found_user = await repository.get_by_telegram_user_id(telegram_user_id=123)

    assert found_user == user
    assert found_user is not None
    assert found_user.telegram_user_id == 123
    assert found_user.username == "andrew"


async def test_user_repository_returns_none_for_unknown_telegram_id(
    session: AsyncSession,
) -> None:
    repository = UserRepository(session=session)

    user = await repository.get_by_telegram_user_id(telegram_user_id=404)

    assert user is None
