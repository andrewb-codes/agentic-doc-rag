from collections.abc import AsyncGenerator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from agentic_rag.api.deps import get_indexing_service
from agentic_rag.api.main import app
from agentic_rag.models import DocumentChunk


class NoOpIndexingService:
    async def index_chunks(self, *, chunks: list[DocumentChunk]) -> None:
        pass


async def get_noop_indexing_service() -> NoOpIndexingService:
    return NoOpIndexingService()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    app.dependency_overrides[get_indexing_service] = get_noop_indexing_service

    try:
        async with (
            LifespanManager(app),
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client,
        ):
            yield client
    finally:
        app.dependency_overrides.clear()
