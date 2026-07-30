import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.fsm.storage.memory import MemoryStorage

from agentic_rag.bot.client import BackendClient
from agentic_rag.bot.handlers import build_router
from agentic_rag.core.config import settings
from agentic_rag.core.logging import configure_logging


async def main() -> None:
    configure_logging(settings)

    if settings.telegram_bot_token is None:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run Telegram bot")

    session = AiohttpSession()
    bot = Bot(token=settings.telegram_bot_token, session=session)

    backend = BackendClient(
        base_url=settings.api_base_url,
        internal_api_key=settings.internal_api_key,
    )
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(build_router(backend=backend))

    try:
        await dispatcher.start_polling(bot)
    finally:
        await backend.close()
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
