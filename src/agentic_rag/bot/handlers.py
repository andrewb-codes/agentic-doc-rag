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

BACK_TO_DOCUMENTS_CALLBACK = "back_to_documents"
SELECT_ALL_DOCUMENTS_CALLBACK = "select_all_documents"
SELECT_ALL_DOCUMENTS_FROM_DETAILS_CALLBACK = "select_all_documents_from_details"
DELETE_DOCUMENT_PREFIX = "delete_document:"
SELECT_DOCUMENT_PREFIX = "select_document:"
VIEW_DOCUMENT_PREFIX = "view_document:"

DocumentPayload = dict[str, Any]
IndexedDocument = tuple[int, DocumentPayload]


def telegram_user_from_aiogram_user(user: User) -> TelegramUser:
    return TelegramUser(
        telegram_user_id=user.id,
        username=user.username,
    )


def telegram_user_from_message(message: Message) -> TelegramUser:
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


def find_document_by_id(
    documents: list[DocumentPayload],
    document_id: int,
) -> IndexedDocument | None:
    return next(
        (
            (index, document)
            for index, document in enumerate(documents, start=1)
            if document["id"] == document_id
        ),
        None,
    )


def documents_list_text(
    documents: list[DocumentPayload],
    active_document_id: int | None,
) -> str:
    lines = [
        f"{index}. {'✅ ' if document['id'] == active_document_id else ''}"
        f"{document['filename']} ({document['status']})"
        for index, document in enumerate(documents, start=1)
    ]

    if active_document_id is None:
        lines.append("\nСейчас выбраны: все документы")
        return "\n".join(lines)

    active_document = find_document_by_id(documents, active_document_id)
    if active_document is not None:
        lines.append(f"\nСейчас выбран: {active_document[1]['filename']}")

    return "\n".join(lines)


def build_documents_list_keyboard(
    documents: list[DocumentPayload],
    active_document_id: int | None,
) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []

    for index, document in enumerate(documents, start=1):
        is_active = document["id"] == active_document_id
        row.append(
            InlineKeyboardButton(
                text=f"{'✅ ' if is_active else ''}{index}",
                callback_data=f"{VIEW_DOCUMENT_PREFIX}{document['id']}",
            )
        )

        if len(row) == 5:
            buttons.append(row)
            row = []

    if row:
        buttons.append(row)

    buttons.append(
        [
            InlineKeyboardButton(
                text="✅ Все документы" if active_document_id is None else "Все документы",
                callback_data=SELECT_ALL_DOCUMENTS_CALLBACK,
            )
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def document_details_text(
    document: DocumentPayload,
    index: int,
    active_document_id: int | None,
) -> str:
    is_active = document["id"] == active_document_id
    created_at = document.get("created_at") or "неизвестно"
    page_count = document.get("page_count") or "неизвестно"
    chunk_count = document.get("chunk_count") or "неизвестно"

    return (
        f"Документ {index}\n\n"
        f"Файл: {document['filename']}\n"
        f"Статус: {document['status']}\n"
        f"Загружен: {created_at}\n"
        f"Страниц: {page_count}\n"
        f"Фрагментов: {chunk_count}\n\n"
        f"{'Сейчас выбран.' if is_active else 'Сейчас не выбран.'}"
    )


def build_document_details_keyboard(
    document: DocumentPayload,
    active_document_id: int | None,
) -> InlineKeyboardMarkup:
    is_active = document["id"] == active_document_id

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="❌ Отметить выбор" if is_active else "✅ Выбрать",
                    callback_data=(
                        SELECT_ALL_DOCUMENTS_FROM_DETAILS_CALLBACK
                        if is_active
                        else f"{SELECT_DOCUMENT_PREFIX}{document['id']}"
                    ),
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Удалить",
                    callback_data=f"{DELETE_DOCUMENT_PREFIX}{document['id']}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data=BACK_TO_DOCUMENTS_CALLBACK,
                )
            ],
        ]
    )


def callback_document_id(callback: CallbackQuery, prefix: str) -> int | None:
    if callback.data is None:
        return None

    return int(callback.data.removeprefix(prefix))


async def active_document_id_from_state(state: FSMContext) -> int | None:
    state_data = await state.get_data()
    active_document_id = state_data.get(ACTIVE_DOCUMENT_ID_KEY)
    return int(cast(str, active_document_id)) if active_document_id is not None else None


async def answer_callback_message(callback: CallbackQuery, text: str) -> None:
    if callback.message is None or isinstance(callback.message, InaccessibleMessage):
        await callback.answer(text)
        return

    await callback.message.answer(text)
    await callback.answer()


async def edit_documents_list_message(
    callback: CallbackQuery,
    documents: list[DocumentPayload],
    active_document_id: int | None,
) -> bool:
    if callback.message is None or isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return False

    if not documents:
        await callback.message.edit_text("Документы пока не загружены.", reply_markup=None)
        return True

    await callback.message.edit_text(
        documents_list_text(documents, active_document_id),
        reply_markup=build_documents_list_keyboard(documents, active_document_id),
    )
    return True


async def edit_document_details_message(
    callback: CallbackQuery,
    document: DocumentPayload,
    index: int,
    active_document_id: int | None,
) -> bool:
    if callback.message is None or isinstance(callback.message, InaccessibleMessage):
        await callback.answer()
        return False

    await callback.message.edit_text(
        document_details_text(document, index, active_document_id),
        reply_markup=build_document_details_keyboard(document, active_document_id),
    )
    return True


async def list_documents_for_callback(
    *,
    backend: BackendClient,
    callback: CallbackQuery,
) -> list[DocumentPayload] | None:
    try:
        return await backend.list_documents(
            user=telegram_user_from_aiogram_user(callback.from_user)
        )
    except httpx.HTTPStatusError as exc:
        await answer_callback_message(callback, backend_error_text(exc))
    except httpx.RequestError:
        await answer_callback_message(callback, backend_request_error_text())

    return None


def build_router(*, backend: BackendClient) -> Router:
    router = Router()

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
            user = telegram_user_from_message(message)
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
            history = await backend.list_history(user=telegram_user_from_message(message))
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
            documents = await backend.list_documents(user=telegram_user_from_message(message))
        except httpx.HTTPStatusError as exc:
            await message.answer(backend_error_text(exc))
            return
        except httpx.RequestError:
            await message.answer(backend_request_error_text())
            return

        if not documents:
            await message.answer("Документы пока не загружены.")
            return

        active_document_id = await active_document_id_from_state(state)
        await message.answer(
            documents_list_text(documents, active_document_id),
            reply_markup=build_documents_list_keyboard(documents, active_document_id),
        )

    @router.callback_query(F.data == SELECT_ALL_DOCUMENTS_CALLBACK)
    async def select_all_documents(callback: CallbackQuery, state: FSMContext) -> None:
        if await active_document_id_from_state(state) is None:
            await callback.answer("Уже выбраны все документы.")
            return

        documents = await list_documents_for_callback(backend=backend, callback=callback)
        if documents is None:
            return

        await state.clear()
        if await edit_documents_list_message(callback, documents, None):
            await callback.answer("Выбраны все документы.")

    @router.callback_query(F.data.startswith(VIEW_DOCUMENT_PREFIX))
    async def view_document(callback: CallbackQuery, state: FSMContext) -> None:
        document_id = callback_document_id(callback, VIEW_DOCUMENT_PREFIX)
        if document_id is None:
            return

        documents = await list_documents_for_callback(backend=backend, callback=callback)
        if documents is None:
            return

        selected = find_document_by_id(documents, document_id)
        if selected is None:
            await answer_callback_message(callback, "Документ не найден.")
            return

        index, document = selected
        active_document_id = await active_document_id_from_state(state)
        if await edit_document_details_message(callback, document, index, active_document_id):
            await callback.answer()

    @router.callback_query(F.data.startswith(SELECT_DOCUMENT_PREFIX))
    async def select_document(callback: CallbackQuery, state: FSMContext) -> None:
        document_id = callback_document_id(callback, SELECT_DOCUMENT_PREFIX)
        if document_id is None:
            return

        documents = await list_documents_for_callback(backend=backend, callback=callback)
        if documents is None:
            return

        selected = find_document_by_id(documents, document_id)
        if selected is None:
            await answer_callback_message(callback, "Документ не найден.")
            return

        current_active_document_id = await active_document_id_from_state(state)
        if current_active_document_id == document_id:
            await callback.answer("Документ уже выбран.")
            return

        index, document = selected
        await state.update_data(
            active_document_id=document["id"],
            active_document_filename=document["filename"],
        )
        if await edit_document_details_message(callback, document, index, document["id"]):
            await callback.answer("Документ выбран.")

    @router.callback_query(F.data == SELECT_ALL_DOCUMENTS_FROM_DETAILS_CALLBACK)
    async def select_all_documents_from_details(callback: CallbackQuery, state: FSMContext) -> None:
        active_document_id = await active_document_id_from_state(state)
        if active_document_id is None:
            await callback.answer("Уже выбраны все документы.")
            return

        documents = await list_documents_for_callback(backend=backend, callback=callback)
        if documents is None:
            return

        selected = find_document_by_id(documents, active_document_id)
        if selected is None:
            await answer_callback_message(callback, "Документ не найден.")
            return

        index, document = selected
        await state.clear()
        if await edit_document_details_message(callback, document, index, None):
            await callback.answer("Выбраны все документы.")

    @router.callback_query(F.data.startswith(DELETE_DOCUMENT_PREFIX))
    async def delete_document(callback: CallbackQuery, state: FSMContext) -> None:
        document_id = callback_document_id(callback, DELETE_DOCUMENT_PREFIX)
        if document_id is None:
            return

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

        active_document_id = await active_document_id_from_state(state)
        if active_document_id == document_id:
            await state.clear()
            active_document_id = None

        if await edit_documents_list_message(callback, documents, active_document_id):
            await callback.answer("Документ удален.")

    @router.callback_query(F.data == BACK_TO_DOCUMENTS_CALLBACK)
    async def back_to_documents(callback: CallbackQuery, state: FSMContext) -> None:
        documents = await list_documents_for_callback(backend=backend, callback=callback)
        if documents is None:
            return

        active_document_id = await active_document_id_from_state(state)
        if await edit_documents_list_message(callback, documents, active_document_id):
            await callback.answer()

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
                user=telegram_user_from_message(message),
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
            user = telegram_user_from_message(message)
            documents = await backend.list_documents(user=user)

            if not documents:
                await status_message.edit_text(
                    "Сначала пришли PDF-документ, затем я смогу отвечать на вопросы."
                )
                return

            active_document_id = await active_document_id_from_state(state)
            if active_document_id is not None:
                result = await backend.ask_document(
                    user=user,
                    document_id=active_document_id,
                    question=message.text,
                )
            else:
                result = await backend.ask_documents(
                    user=user,
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
