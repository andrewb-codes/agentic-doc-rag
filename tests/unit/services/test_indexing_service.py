from unittest.mock import AsyncMock

import pytest

from agentic_rag.models import DocumentChunk
from agentic_rag.services.embedding import FakeEmbeddingService
from agentic_rag.services.indexing import DocumentIndexingService

pytestmark = pytest.mark.no_db


async def test_indexing_service_embeds_and_upserts_chunks() -> None:
    embedding_service = FakeEmbeddingService(vector_size=3)
    vector_store = AsyncMock()

    service = DocumentIndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    chunks = [
        DocumentChunk(
            id=1,
            document_id=10,
            page=1,
            chunk_index=0,
            text="first chunk",
            source="manual.pdf",
        ),
        DocumentChunk(
            id=2,
            document_id=10,
            page=1,
            chunk_index=1,
            text="second chunk",
            source="manual.pdf",
        ),
    ]

    await service.index_chunks(chunks=chunks)

    vector_store.ensure_collection.assert_awaited_once_with(vector_size=3)
    vector_store.upsert_chunks.assert_awaited_once_with(
        chunks=chunks,
        embeddings=[
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
    )
