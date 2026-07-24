from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import UnauthorizedError
from agentic_rag.db.session import get_db_session
from agentic_rag.models import User
from agentic_rag.services.document import DocumentService
from agentic_rag.services.embedding import EmbeddingService, OpenAIEmbeddingService
from agentic_rag.services.indexing import DocumentIndexingService
from agentic_rag.services.user import UserService
from agentic_rag.vectorstores.qdrant import QdrantVectorStore


def verify_internal_api_key(
    x_internal_api_key: Annotated[str | None, Header(alias="X-Internal-Api-Key")] = None,
) -> None:
    if x_internal_api_key != settings.internal_api_key:
        raise UnauthorizedError()


def get_session(session: Annotated[AsyncSession, Depends(get_db_session)]) -> AsyncSession:
    return session


def get_user_service(session: Annotated[AsyncSession, Depends(get_session)]) -> UserService:
    return UserService(session=session)


def get_embedding_service() -> EmbeddingService:
    return OpenAIEmbeddingService(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )


async def get_vector_store() -> AsyncGenerator[QdrantVectorStore]:
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )

    try:
        yield vector_store
    finally:
        await vector_store.close()


def get_indexing_service(
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    vector_store: Annotated[QdrantVectorStore, Depends(get_vector_store)],
) -> DocumentIndexingService:
    return DocumentIndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def get_document_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    indexing_service: Annotated[DocumentIndexingService, Depends(get_indexing_service)],
) -> DocumentService:
    return DocumentService(
        session=session,
        indexing_service=indexing_service,
    )


async def get_current_telegram_user(
    service: Annotated[UserService, Depends(get_user_service)],
    _: Annotated[None, Depends(verify_internal_api_key)],
    telegram_user_id: Annotated[int | None, Header(alias="X-Telegram-User-Id")] = None,
    telegram_username: Annotated[str | None, Header(alias="X-Telegram-Username")] = None,
) -> User:
    if telegram_user_id is None or telegram_user_id <= 0:
        raise UnauthorizedError()

    return await service.get_or_create_telegram_user(
        telegram_user_id=telegram_user_id,
        telegram_username=telegram_username,
    )
