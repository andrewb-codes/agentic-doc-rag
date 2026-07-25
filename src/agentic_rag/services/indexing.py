from typing import Protocol

from agentic_rag.models import DocumentChunk
from agentic_rag.services.embedding import EmbeddingService
from agentic_rag.vectorstores.qdrant import QdrantVectorStore


class IndexingService(Protocol):
    async def index_chunks(self, *, chunks: list[DocumentChunk]) -> None:
        pass


class DocumentIndexingService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def index_chunks(self, *, chunks: list[DocumentChunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.embed_texts(texts=texts)

        await self.vector_store.ensure_collection(
            vector_size=self.embedding_service.vector_size,
        )
        await self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )
