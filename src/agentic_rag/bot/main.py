import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.fsm.storage.memory import MemoryStorage

from agentic_rag.bot.client import BackendClient
from agentic_rag.bot.config import bot_settings
from agentic_rag.bot.handlers import build_router

logger = logging.getLogger(__name__)


def configure_bot_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


async def main() -> None:
    configure_bot_logging()

    session = AiohttpSession()
    bot = Bot(token=bot_settings.telegram_bot_token, session=session)

    backend = BackendClient(
        base_url=bot_settings.api_base_url,
        internal_api_key=bot_settings.internal_api_key,
    )
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(build_router(backend=backend))

    try:
        logger.info("Telegram bot polling started")
        await dispatcher.start_polling(bot)
    finally:
        await backend.close()
        await session.close()
        logger.info("Telegram bot polling stopped")


if __name__ == "__main__":
    asyncio.run(main())
