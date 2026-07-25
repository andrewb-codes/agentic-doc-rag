from dataclasses import dataclass
from typing import Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from agentic_rag.models import DocumentChunk


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: int
    score: float


class VectorSizeMismatchError(Exception):
    pass


class QdrantClient(Protocol):
    async def get_collections(self) -> object:
        pass

    async def get_collection(self, *, collection_name: str) -> object:
        pass

    async def create_collection(
        self, *, collection_name: str, vectors_config: VectorParams
    ) -> object:
        pass

    async def upsert(self, *, collection_name: str, points: list[PointStruct]) -> object:
        pass

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        query_filter: Filter,
        limit: int,
    ) -> object:
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
        self,
        *,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        owner_id: int,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "owner_id": owner_id,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
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

    async def search_chunks(
        self,
        *,
        embedding: list[float],
        owner_id: int,
        limit: int,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        filter_conditions = [
            FieldCondition(
                key="owner_id",
                match=MatchValue(value=owner_id),
            )
        ]

        if document_id is not None:
            filter_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        query_filter = Filter(must=filter_conditions)

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            VectorSearchResult(
                chunk_id=int(point.payload["chunk_id"]),
                score=float(point.score),
            )
            for point in response.points
        ]

    async def close(self) -> None:
        await self.client.close()
