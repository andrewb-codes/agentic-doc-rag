from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from agentic_rag.models import DocumentChunk
from agentic_rag.repositories.chunk import DocumentChunkRepository
from agentic_rag.services.embedding import FakeEmbeddingService
from agentic_rag.services.retrieval import RetrievalService
from agentic_rag.vectorstores.qdrant import QdrantVectorStore, VectorSearchResult

pytestmark = pytest.mark.no_db


async def test_retrieval_service_embeds_query_searches_qdrant_and_loads_chunks() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=1,
        chunk_index=0,
        text="chunk text",
        source="manual.pdf",
    )

    get_by_ids = AsyncMock(return_value=[chunk])
    search_chunks = AsyncMock(return_value=[VectorSearchResult(chunk_id=10, score=0.91)])

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(vector_size=3),
        chunk_repository=cast(
            DocumentChunkRepository,
            cast(object, SimpleNamespace(get_by_ids=get_by_ids)),
        ),
        vector_store=cast(
            QdrantVectorStore,
            cast(object, SimpleNamespace(search_chunks=search_chunks)),
        ),
    )

    chunks = await service.search_user_chunks(
        query="question",
        owner_id=1,
        limit=5,
    )

    search_chunks.assert_awaited_once_with(
        embedding=[1.0, 1.0, 1.0],
        owner_id=1,
        document_id=None,
        limit=5,
    )
    get_by_ids.assert_awaited_once_with(chunk_ids=[10])
    assert chunks == [chunk]


async def test_retrieval_service_searches_chunks_inside_document() -> None:
    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=1,
        chunk_index=0,
        text="chunk text",
        source="manual.pdf",
    )

    get_by_ids = AsyncMock(return_value=[chunk])
    search_chunks = AsyncMock(return_value=[VectorSearchResult(chunk_id=10, score=0.91)])

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(vector_size=3),
        chunk_repository=cast(
            DocumentChunkRepository,
            cast(object, SimpleNamespace(get_by_ids=get_by_ids)),
        ),
        vector_store=cast(
            QdrantVectorStore,
            cast(object, SimpleNamespace(search_chunks=search_chunks)),
        ),
    )

    chunks = await service.search_document_chunks(
        query="question",
        owner_id=1,
        document_id=20,
        limit=5,
    )

    search_chunks.assert_awaited_once_with(
        embedding=[1.0, 1.0, 1.0],
        owner_id=1,
        document_id=20,
        limit=5,
    )
    get_by_ids.assert_awaited_once_with(chunk_ids=[10])
    assert chunks == [chunk]
