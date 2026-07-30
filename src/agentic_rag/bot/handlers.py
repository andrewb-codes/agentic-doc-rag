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

HELP_TEXT = (
    "Пришли PDF-документ, я обработаю его и смогу отвечать на вопросы.\n\n"
    "📄 Документы – список, выбор активного документа и удаление.\n"
    "📜 История – последние вопросы и ответы.\n"
    "📊 Статус – количество сохраненных документов и вопросов.\n\n"
    "Если выбран активный документ, вопросы идут только по нему. "
    "Если выбраны все документы, поиск идет по всем загруженным."
)

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
DELETE_DOCUMENT_PREFIX = "delete_document:"


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

    def backend_request_error_text() -> str:
        return "Backend сейчас недоступен. Попробуй позже."

    def build_documents_keyboard(
        documents: list[dict[str, Any]],
        active_document_id: int | None,
    ) -> InlineKeyboardMarkup:
        buttons = []
        for index, document in enumerate(documents, start=1):
            is_active = document["id"] == active_document_id
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=f"{'✅ Выбран' if is_active else 'Выбрать'} {index}",
                        callback_data=f"{SELECT_DOCUMENT_PREFIX}{document['id']}",
                    ),
                    InlineKeyboardButton(
                        text=f"Удалить {index}",
                        callback_data=f"{DELETE_DOCUMENT_PREFIX}{document['id']}",
                    ),
                ]
            )

        buttons.append(
            [
                InlineKeyboardButton(
                    text="✅ Все документы" if active_document_id is None else "Все документы",
                    callback_data=CLEAR_DOCUMENT_CALLBACK,
                )
            ]
        )

        return InlineKeyboardMarkup(inline_keyboard=buttons)

    def documents_list_text(
        documents: list[dict[str, Any]],
        active_document_id: int | None,
    ) -> str:
        lines = [
            f"{index}. {'✅ ' if document['id'] == active_document_id else ''}"
            f"{document['filename']} ({document['status']})"
            for index, document in enumerate(documents, start=1)
        ]

        if active_document_id is None:
            lines.append("\nСейчас выбраны: все документы")
        else:
            active_document = next(
                (document for document in documents if document["id"] == active_document_id),
                None,
            )
            if active_document is not None:
                lines.append(f"\nСейчас выбран: {active_document['filename']}")

        return "\n".join(lines)

    async def answer_callback_message(callback: CallbackQuery, text: str) -> None:
        if callback.message is None or isinstance(callback.message, InaccessibleMessage):
            await callback.answer(text)
            return

        await callback.message.answer(text)
        await callback.answer()

    async def edit_documents_message(
        callback: CallbackQuery,
        documents: list[dict[str, Any]],
        active_document_id: int | None,
    ) -> None:
        if callback.message is None or isinstance(callback.message, InaccessibleMessage):
            await callback.answer()
            return

        await callback.message.edit_text(
            documents_list_text(documents, active_document_id),
            reply_markup=build_documents_keyboard(documents, active_document_id),
        )

    async def edit_or_clear_documents_message(
        callback: CallbackQuery,
        documents: list[dict[str, Any]],
        active_document_id: int | None,
    ) -> None:
        if callback.message is None or isinstance(callback.message, InaccessibleMessage):
            await callback.answer()
            return

        if not documents:
            await callback.message.edit_text("Документы пока не загружены.", reply_markup=None)
            return

        await edit_documents_message(callback, documents, active_document_id)

    @router.message(CommandStart())
    async def start_command(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=MAIN_KEYBOARD)

    @router.message(Command("help"))
    @router.message(F.text == HELP_BUTTON)
    async def help_command(message: Message) -> None:
        await message.answer(HELP_TEXT, reply_markup=MAIN_KEYBOARD)

    @router.message(Command("status"))
    @router.message(F.text == STATUS_BUTTON)
    async def status_command(message: Message, state: FSMContext) -> None:
        try:
            user = current_user(message)
            documents = await backend.list_documents(user=user)
            history = await backend.list_history(user=user)
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return
        except httpx.RequestError:
            await message.answer(backend_request_error_text())
            return

        state_data = await state.get_data()
        active_document_filename = state_data.get(ACTIVE_DOCUMENT_FILENAME_KEY)

        active_scope = (
            f"Активный документ: {active_document_filename}"
            if active_document_filename is not None
            else "Активный режим: все документы"
        )

        await message.answer(
            f"Документов: {len(documents)}\nВопросов в истории: {len(history)}\n\n{active_scope}"
        )

    @router.message(Command("history"))
    @router.message(F.text == HISTORY_BUTTON)
    async def list_history(message: Message) -> None:
        try:
            history = await backend.list_history(user=current_user(message))
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return
        except httpx.RequestError:
            await message.answer(backend_request_error_text())
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
    async def list_documents(message: Message, state: FSMContext) -> None:
        try:
            documents = await backend.list_documents(user=current_user(message))
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return
        except httpx.RequestError:
            await message.answer(backend_request_error_text())
            return

        if not documents:
            await message.answer("Документы пока не загружены.")
            return

        state_data = await state.get_data()
        active_document_id = state_data.get(ACTIVE_DOCUMENT_ID_KEY)
        active_document_id = (
            int(cast(str, active_document_id)) if active_document_id is not None else None
        )

        await message.answer(
            documents_list_text(documents, active_document_id),
            reply_markup=build_documents_keyboard(documents, active_document_id),
        )

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
        except httpx.RequestError:
            await answer_callback_message(callback, backend_request_error_text())
            return

        selected_document = next(
            (document for document in documents if document["id"] == document_id),
            None,
        )

        if selected_document is None:
            await answer_callback_message(callback, "Документ не найден.")
            return

        state_data = await state.get_data()
        current_active_document_id = state_data.get(ACTIVE_DOCUMENT_ID_KEY)

        if (
            current_active_document_id is not None
            and int(cast(str, current_active_document_id)) == document_id
        ):
            await callback.answer("Документ уже выбран.")
            return

        await state.update_data(
            active_document_id=selected_document["id"],
            active_document_filename=selected_document["filename"],
        )

        await edit_documents_message(callback, documents, selected_document["id"])
        await callback.answer("Документ выбран.")

    @router.callback_query(F.data == CLEAR_DOCUMENT_CALLBACK)
    async def clear_document(callback: CallbackQuery, state: FSMContext) -> None:
        state_data = await state.get_data()
        active_document_id = state_data.get(ACTIVE_DOCUMENT_ID_KEY)

        if active_document_id is None:
            await callback.answer("Уже выбраны все документы.")
            return

        user = telegram_user_from_aiogram_user(callback.from_user)

        try:
            documents = await backend.list_documents(user=user)
        except httpx.HTTPStatusError as exc:
            await answer_callback_message(callback, backend_error_text(exc))
            return
        except httpx.RequestError:
            await answer_callback_message(callback, backend_request_error_text())
            return

        await state.clear()
        await edit_documents_message(callback, documents, None)
        await callback.answer("Выбраны все документы.")

    @router.callback_query(F.data.startswith(DELETE_DOCUMENT_PREFIX))
    async def delete_document(callback: CallbackQuery, state: FSMContext) -> None:
        if callback.data is None:
            return

        document_id = int(callback.data.removeprefix(DELETE_DOCUMENT_PREFIX))
        user = telegram_user_from_aiogram_user(callback.from_user)

        try:
            await backend.delete_document(user=user, document_id=document_id)
            documents = await backend.list_documents(user=user)
        except httpx.HTTPStatusError as exc:
            await answer_callback_message(callback, backend_error_text(exc))
            return
        except httpx.RequestError:
            await answer_callback_message(callback, backend_request_error_text())
            return

        state_data = await state.get_data()
        active_document_id = state_data.get(ACTIVE_DOCUMENT_ID_KEY)
        active_document_id = (
            int(cast(str, active_document_id)) if active_document_id is not None else None
        )

        if active_document_id == document_id:
            await state.clear()
            active_document_id = None

        await edit_or_clear_documents_message(callback, documents, active_document_id)
        await callback.answer("Документ удален.")

    @router.message(F.document)
    async def upload_document(message: Message, bot: Bot, state: FSMContext) -> None:
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
        except httpx.RequestError:
            await status_message.edit_text(backend_request_error_text())
            return

        await state.update_data(
            active_document_id=result["id"],
            active_document_filename=result["filename"],
        )

        await status_message.edit_text(
            f"Документ готов: {result['filename']}\n"
            f"Страниц: {result['page_count']}\n"
            f"Фрагментов: {result['chunk_count']}\n\n"
            "Я выбрал этот документ активным. Теперь можешь задавать вопросы по нему."
        )

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
        except httpx.RequestError:
            await status_message.edit_text(backend_request_error_text())
            return

        await status_message.edit_text(result["answer"])

    return router
