from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue

from agentic_rag.models import DocumentChunk
from agentic_rag.vectorstores.qdrant import QdrantVectorStore, VectorSizeMismatchError

pytestmark = pytest.mark.no_db


async def test_qdrant_healthcheck_returns_true_when_client_responds() -> None:
    client = AsyncMock()
    client.get_collections = AsyncMock()

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    assert await vector_store.healthcheck() is True


async def test_qdrant_healthcheck_returns_false_when_client_fails() -> None:
    client = AsyncMock()
    client.get_collections = AsyncMock(side_effect=RuntimeError("qdrant down"))

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    assert await vector_store.healthcheck() is False


async def test_qdrant_ensure_collection_does_not_create_existing_collection() -> None:
    client = AsyncMock()
    client.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[SimpleNamespace(name="document_chunks")])
    )
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=1536),
                )
            )
        )
    )
    client.create_collection = AsyncMock()

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    await vector_store.ensure_collection(vector_size=1536)

    client.get_collection.assert_awaited_once_with(collection_name="document_chunks")
    client.create_collection.assert_not_awaited()


async def test_qdrant_ensure_collection_rejects_vector_size_mismatch() -> None:
    client = AsyncMock()
    client.get_collections = AsyncMock(
        return_value=SimpleNamespace(collections=[SimpleNamespace(name="document_chunks")])
    )
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=3),
                )
            )
        )
    )

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    with pytest.raises(VectorSizeMismatchError, match="has vector size 3, expected 1536"):
        await vector_store.ensure_collection(vector_size=1536)


async def test_qdrant_ensure_collection_creates_missing_collection() -> None:
    client = AsyncMock()
    client.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
    client.create_collection = AsyncMock()

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    await vector_store.ensure_collection(vector_size=1536)

    client.create_collection.assert_awaited_once()
    kwargs = client.create_collection.await_args.kwargs

    assert kwargs["collection_name"] == "document_chunks"
    assert kwargs["vectors_config"].size == 1536
    assert kwargs["vectors_config"].distance == Distance.COSINE


async def test_qdrant_upsert_chunks_sends_points_with_payload() -> None:
    client = AsyncMock()
    client.upsert = AsyncMock()

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="chunk text",
        source="manual.pdf",
    )

    await vector_store.upsert_chunks(
        chunks=[chunk],
        embeddings=[[0.1, 0.2, 0.3]],
        owner_id=1,
    )

    client.upsert.assert_awaited_once()
    kwargs = client.upsert.await_args.kwargs

    assert kwargs["collection_name"] == "document_chunks"

    point = kwargs["points"][0]
    assert point.id == 10
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {
        "owner_id": 1,
        "document_id": 20,
        "chunk_id": 10,
        "page": 3,
        "chunk_index": 0,
        "source": "manual.pdf",
    }


async def test_qdrant_upsert_chunks_rejects_mismatched_embeddings_count() -> None:
    client = AsyncMock()

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    chunk = DocumentChunk(
        id=10,
        document_id=20,
        page=3,
        chunk_index=0,
        text="chunk text",
        source="manual.pdf",
    )

    with pytest.raises(ValueError, match="same length"):
        await vector_store.upsert_chunks(
            chunks=[chunk],
            embeddings=[],
            owner_id=1,
        )


async def test_qdrant_search_chunks_returns_chunk_ids_and_scores() -> None:
    client = AsyncMock()
    client.query_points = AsyncMock(
        return_value=SimpleNamespace(
            points=[
                SimpleNamespace(payload={"chunk_id": 10}, score=0.91),
                SimpleNamespace(payload={"chunk_id": 20}, score=0.82),
            ]
        )
    )

    vector_store = QdrantVectorStore(
        url="http://unused",
        collection_name="document_chunks",
        client=client,
    )

    results = await vector_store.search_chunks(
        embedding=[0.1, 0.2, 0.3],
        owner_id=1,
        limit=2,
    )

    client.query_points.assert_awaited_once_with(
        collection_name="document_chunks",
        query=[0.1, 0.2, 0.3],
        limit=2,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="owner_id",
                    match=MatchValue(value=1),
                )
            ]
        ),
    )
    assert [(result.chunk_id, result.score) for result in results] == [
        (10, 0.91),
        (20, 0.82),
    ]
