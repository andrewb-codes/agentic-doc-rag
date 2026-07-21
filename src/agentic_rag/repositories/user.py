from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_telegram_user_id(self, *, telegram_user_id: int) -> User | None:
        query = select(User).where(User.telegram_user_id == telegram_user_id)
        return cast(User | None, await self.session.scalar(query))

    async def create(self, *, telegram_user_id: int, username: str | None) -> User:
        user = User(telegram_user_id=telegram_user_id, username=username)
        self.session.add(user)
        await self.session.flush()
        return user
