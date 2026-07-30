from io import BytesIO
from typing import Any, cast

import httpx
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InaccessibleMessage,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    User,
)

from agentic_rag.bot.client import BackendClient, TelegramUser

HELP_TEXT = "Пришли PDF-документ, я обработаю его и смогу отвечать на вопросы по содержимому."

DOCUMENTS_BUTTON = "📄 Документы"
HISTORY_BUTTON = "📜 История"
STATUS_BUTTON = "📊 Статус"
HELP_BUTTON = "ℹ️ Как пользоваться"

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text=DOCUMENTS_BUTTON),
            KeyboardButton(text=HISTORY_BUTTON),
        ],
        [
            KeyboardButton(text=STATUS_BUTTON),
            KeyboardButton(text=HELP_BUTTON),
        ],
    ],
    resize_keyboard=True,
)

ACTIVE_DOCUMENT_ID_KEY = "active_document_id"
ACTIVE_DOCUMENT_FILENAME_KEY = "active_document_filename"
SELECT_DOCUMENT_PREFIX = "select_document:"
CLEAR_DOCUMENT_CALLBACK = "clear_document"


def build_router(*, backend: BackendClient) -> Router:
    router = Router()

    def telegram_user_from_aiogram_user(user: User) -> TelegramUser:
        return TelegramUser(
            telegram_user_id=user.id,
            username=user.username,
        )

    def current_user(message: Message) -> TelegramUser:
        if message.from_user is None:
            raise RuntimeError("telegram user is missing")

        return telegram_user_from_aiogram_user(message.from_user)

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

    def build_documents_keyboard(documents: list[dict[str, Any]]) -> InlineKeyboardMarkup:
        buttons = [
            [
                InlineKeyboardButton(
                    text=f"Выбрать {index}",
                    callback_data=f"{SELECT_DOCUMENT_PREFIX}{document['id']}",
                )
            ]
            for index, document in enumerate(documents, start=1)
        ]
        buttons.append(
            [
                InlineKeyboardButton(
                    text="Все документы",
                    callback_data=CLEAR_DOCUMENT_CALLBACK,
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=buttons)

    async def answer_callback_message(callback: CallbackQuery, text: str) -> None:
        if callback.message is None or isinstance(callback.message, InaccessibleMessage):
            await callback.answer(text)
            return

        await callback.message.answer(text)
        await callback.answer()

    @router.message(CommandStart())
    async def start_command(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=MAIN_KEYBOARD)

    @router.message(Command("help"))
    @router.message(F.text == HELP_BUTTON)
    async def help_command(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=MAIN_KEYBOARD)

    @router.message(Command("status"))
    @router.message(F.text == STATUS_BUTTON)
    async def status_command(message: Message) -> None:
        try:
            user = current_user(message)
            documents = await backend.list_documents(user=user)
            history = await backend.list_history(user=user)
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return

        await message.answer(
            f"Документов: {len(documents)}\nВопросов в истории: {len(history)}\n\n"
        )

    @router.message(Command("history"))
    @router.message(F.text == HISTORY_BUTTON)
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

    @router.message(Command("documents"))
    @router.message(F.text == DOCUMENTS_BUTTON)
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
            f"{index}. {document['filename']} ({document['status']})"
            for index, document in enumerate(documents, start=1)
        )
        await message.answer(text, reply_markup=build_documents_keyboard(documents))

    @router.callback_query(F.data.startswith(SELECT_DOCUMENT_PREFIX))
    async def select_document(callback: CallbackQuery, state: FSMContext) -> None:
        if callback.data is None:
            return

        document_id = int(callback.data.removeprefix(SELECT_DOCUMENT_PREFIX))
        user = telegram_user_from_aiogram_user(callback.from_user)

        try:
            documents = await backend.list_documents(user=user)
        except httpx.HTTPStatusError as exc:
            await answer_callback_message(callback, backend_error_text(exc))
            return

        selected_document = next(
            (document for document in documents if document["id"] == document_id),
            None,
        )

        if selected_document is None:
            await answer_callback_message(callback, "Документ не найден.")
            return

        await state.update_data(
            active_document_id=selected_document["id"],
            active_document_filename=selected_document["filename"],
        )

        await answer_callback_message(
            callback,
            f"Активный документ: {selected_document['filename']}\n"
            "Теперь вопросы будут идти только по нему.",
        )

    @router.callback_query(F.data == CLEAR_DOCUMENT_CALLBACK)
    async def clear_document(callback: CallbackQuery, state: FSMContext) -> None:
        await state.clear()

        await answer_callback_message(
            callback,
            "Активный документ сброшен. Вопросы будут идти по всем документам.",
        )

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
        await message.answer("Выбери действие или задай вопрос.", reply_markup=MAIN_KEYBOARD)

    @router.message(F.text)
    async def ask_documents(message: Message, state: FSMContext) -> None:
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

            state_data = await state.get_data()
            active_document_id = state_data.get(ACTIVE_DOCUMENT_ID_KEY)

            if active_document_id is not None:
                result = await backend.ask_document(
                    user=current_user(message),
                    document_id=int(cast(str, active_document_id)),
                    question=message.text,
                )
            else:
                result = await backend.ask_documents(
                    user=current_user(message),
                    question=message.text,
                )
        except httpx.HTTPStatusError as exc:
            await status_message.edit_text(backend_error_text(exc))
            return

        await status_message.edit_text(result["answer"])

    return router
