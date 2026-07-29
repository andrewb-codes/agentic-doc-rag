from io import BytesIO

import httpx
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from agentic_rag.bot.client import BackendClient, TelegramUser

HELP_TEXT = (
    "Пришли PDF-документ, я обработаю его и смогу отвечать на вопросы по содержимому.\n\n"
    "Команды:\n"
    "/documents — список загруженных документов\n"
    "/history — последние вопросы и ответы\n"
    "/status — текущее состояние\n"
    "/help — справка"
)


def build_router(*, backend: BackendClient) -> Router:
    router = Router()

    def current_user(message: Message) -> TelegramUser:
        if message.from_user is None:
            raise RuntimeError("telegram user is missing")

        return TelegramUser(
            telegram_user_id=message.from_user.id,
            username=message.from_user.username,
        )

    def backend_error_text(error: httpx.HTTPStatusError) -> str:
        status_code = error.response.status_code

        if status_code == 429:
            return "Слишком много запросов. Попробуй позже."
        if status_code == 413:
            return "Файл слишком большой."
        if status_code == 422:
            return "Не удалось обработать PDF."
        if status_code in {502, 504}:
            return "Внешний AI-провайдер сейчас недоступен. Попробуй позже."

        return "Не удалось выполнить запрос."

    @router.message(CommandStart())
    async def start_command(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @router.message(Command("status"))
    async def status_command(message: Message) -> None:
        try:
            user = current_user(message)
            documents = await backend.list_documents(user=user)
            history = await backend.list_history(user=user)
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return

        await message.answer(
            f"Документов: {len(documents)}\n"
            f"Вопросов в истории: {len(history)}\n\n"
            "Пришли PDF или задай вопрос по загруженным документам."
        )

    @router.message(Command("documents"))
    async def list_documents(message: Message) -> None:
        try:
            documents = await backend.list_documents(user=current_user(message))
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return

        if not documents:
            await message.answer("Документы пока не загружены.")
            return

        text = "\n".join(
            f"{document['id']}. {document['filename']} ({document['status']})"
            for document in documents
        )
        await message.answer(text)

    @router.message(Command("history"))
    async def list_history(message: Message) -> None:
        try:
            history = await backend.list_history(user=current_user(message))
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return

        if not history:
            await message.answer("История вопросов пока пустая.")
            return

        text = "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in reversed(history[:5])
        )
        await message.answer(text)

    @router.message(F.document)
    async def upload_document(message: Message, bot: Bot) -> None:
        document = message.document

        if document is None:
            return

        filename = document.file_name or "document.pdf"

        if not filename.lower().endswith(".pdf"):
            await message.answer("Загружать можно только PDF.")
            return

        file = await bot.get_file(document.file_id)

        if file.file_path is None:
            await message.answer("Не удалось скачать файл.")
            return

        buffer = BytesIO()
        await bot.download_file(file.file_path, destination=buffer)

        status_message = await message.answer(
            "Загружаю и обрабатываю PDF. Это может занять до минуты."
        )

        try:
            result = await backend.upload_document(
                user=current_user(message),
                filename=filename,
                content=buffer.getvalue(),
            )
        except httpx.HTTPStatusError as exc:
            await status_message.edit_text(backend_error_text(exc))
            return

        await status_message.edit_text(
            f"Документ готов: {result['filename']}\n"
            f"Страниц: {result['page_count']}\n"
            f"Фрагментов: {result['chunk_count']}\n\n"
            "Теперь можешь задать вопрос по документу."
        )

    @router.message(F.text)
    async def ask_documents(message: Message) -> None:
        if not message.text:
            return

        status_message = await message.answer("Ищу релевантные фрагменты и готовлю ответ.")

        try:
            documents = await backend.list_documents(user=current_user(message))

            if not documents:
                await status_message.edit_text(
                    "Сначала пришли PDF-документ, затем я смогу отвечать на вопросы."
                )
                return

            result = await backend.ask_documents(
                user=current_user(message),
                question=message.text,
            )
        except httpx.HTTPStatusError as exc:
            await status_message.edit_text(backend_error_text(exc))
            return

        await status_message.edit_text(result["answer"])

    return router
