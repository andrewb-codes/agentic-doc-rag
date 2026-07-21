from collections.abc import AsyncGenerator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from agentic_rag.api.main import app
from agentic_rag.db.session import AsyncSessionLocal


@pytest.fixture(autouse=True)
async def clean_db(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("no_db"):
        return

    async with AsyncSessionLocal() as session:
        await session.execute(
            text("TRUNCATE TABLE qa_history, documents, users RESTART IDENTITY CASCADE")
        )
        await session.commit()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    async with (
        LifespanManager(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        yield client
