from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import UnauthorizedError
from agentic_rag.db.session import get_db_session
from agentic_rag.models import User
from agentic_rag.repositories.chunk import DocumentChunkRepository
from agentic_rag.repositories.document import DocumentRepository
from agentic_rag.repositories.qa_history import QAHistoryRepository
from agentic_rag.repositories.user import UserRepository
from agentic_rag.services.answer import AnswerService
from agentic_rag.services.document import DocumentMetadataService, DocumentProcessingService
from agentic_rag.services.embedding import EmbeddingService, OpenAIEmbeddingService
from agentic_rag.services.indexing import DocumentIndexingService
from agentic_rag.services.llm import OpenAIChatService
from agentic_rag.services.qa_history import QAHistoryService
from agentic_rag.services.retrieval import RetrievalService
from agentic_rag.services.user import UserService
from agentic_rag.vectorstores.qdrant import QdrantVectorStore


def verify_internal_api_key(
    x_internal_api_key: Annotated[str | None, Header(alias="X-Internal-Api-Key")] = None,
) -> None:
    if x_internal_api_key != settings.internal_api_key:
        raise UnauthorizedError()


def get_session(session: Annotated[AsyncSession, Depends(get_db_session)]) -> AsyncSession:
    return session


def get_user_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserRepository:
    return UserRepository(session=session)


def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DocumentRepository:
    return DocumentRepository(session=session)


def get_chunk_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DocumentChunkRepository:
    return DocumentChunkRepository(session=session)


def get_qa_history_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> QAHistoryRepository:
    return QAHistoryRepository(session=session)


def get_user_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    repository: Annotated[UserRepository, Depends(get_user_repository)],
) -> UserService:
    return UserService(session=session, repository=repository)


def get_embedding_service() -> EmbeddingService:
    return OpenAIEmbeddingService(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
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


def get_document_metadata_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    document_repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> DocumentMetadataService:
    return DocumentMetadataService(
        session=session,
        document_repository=document_repository,
    )


def get_document_processing_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    document_repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    chunk_repository: Annotated[DocumentChunkRepository, Depends(get_chunk_repository)],
    indexing_service: Annotated[DocumentIndexingService, Depends(get_indexing_service)],
) -> DocumentProcessingService:
    return DocumentProcessingService(
        session=session,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        indexing_service=indexing_service,
    )


def get_retrieval_service(
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    chunk_repository: Annotated[DocumentChunkRepository, Depends(get_chunk_repository)],
    vector_store: Annotated[QdrantVectorStore, Depends(get_vector_store)],
) -> RetrievalService:
    return RetrievalService(
        embedding_service=embedding_service,
        chunk_repository=chunk_repository,
        vector_store=vector_store,
    )


def get_chat_service() -> OpenAIChatService:
    return OpenAIChatService(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.llm_base_url,
    )


def get_answer_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    chat_service: Annotated[OpenAIChatService, Depends(get_chat_service)],
    qa_history_repository: Annotated[QAHistoryRepository, Depends(get_qa_history_repository)],
) -> AnswerService:
    return AnswerService(
        session=session,
        retrieval_service=retrieval_service,
        chat_service=chat_service,
        qa_history_repository=qa_history_repository,
    )


def get_qa_history_service(
    repository: Annotated[QAHistoryRepository, Depends(get_qa_history_repository)],
) -> QAHistoryService:
    return QAHistoryService(repository=repository)


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
