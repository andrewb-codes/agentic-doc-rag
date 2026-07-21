from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import User
from agentic_rag.repositories.user import UserRepository


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = UserRepository(session)

    async def get_or_create_telegram_user(
        self, *, telegram_user_id: int, telegram_username: str | None
    ) -> User:
        user = await self.repository.get_by_telegram_user_id(telegram_user_id=telegram_user_id)

        if user is not None:
            if telegram_username is not None and user.username != telegram_username:
                user.username = telegram_username

            await self.session.commit()
            await self.session.refresh(user)
            return user

        user = await self.repository.create(
            telegram_user_id=telegram_user_id, username=telegram_username
        )
        await self.session.commit()
        await self.session.refresh(user)
        return user
