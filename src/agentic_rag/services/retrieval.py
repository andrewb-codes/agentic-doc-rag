from agentic_rag.models import DocumentChunk
from agentic_rag.repositories.chunk import DocumentChunkRepository
from agentic_rag.services.embedding import EmbeddingService
from agentic_rag.vectorstores.qdrant import QdrantVectorStore


class RetrievalService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        chunk_repository: DocumentChunkRepository,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository
        self.vector_store = vector_store

    async def search_user_chunks(
        self,
        *,
        query: str,
        owner_id: int,
        limit: int,
    ) -> list[DocumentChunk]:
        embeddings = await self.embedding_service.embed_texts(texts=[query])
        search_results = await self.vector_store.search_chunks(
            embedding=embeddings[0],
            owner_id=owner_id,
            limit=limit,
        )

        chunk_ids = [result.chunk_id for result in search_results]
        return await self.chunk_repository.get_by_ids(chunk_ids=chunk_ids)
