from typing import Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from agentic_rag.models import DocumentChunk


class VectorSizeMismatchError(Exception):
    pass


class QdrantClient(Protocol):
    async def get_collections(self) -> object:
        pass

    async def create_collection(
        self, *, collection_name: str, vectors_config: VectorParams
    ) -> object:
        pass

    async def upsert(self, *, collection_name: str, points: list[PointStruct]) -> object:
        pass

    async def close(self) -> None:
        pass


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        client: QdrantClient | None = None,
    ) -> None:
        self.client = client if client is not None else AsyncQdrantClient(url=url)
        self.collection_name = collection_name

    async def healthcheck(self) -> bool:
        try:
            await self.client.get_collections()
        except Exception:
            return False

        return True

    async def ensure_collection(self, *, vector_size: int) -> None:
        collections_response = await self.client.get_collections()
        collection_names = {collection.name for collection in collections_response.collections}

        if self.collection_name in collection_names:
            collection = await self.client.get_collection(collection_name=self.collection_name)
            existing_vector_size = collection.config.params.vectors.size

            if existing_vector_size != vector_size:
                raise VectorSizeMismatchError(
                    f"qdrant collection '{self.collection_name}' has vector size "
                    f"{existing_vector_size}, expected {vector_size}"
                )

            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    async def upsert_chunks(
        self, *, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.source,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def close(self) -> None:
        await self.client.close()
