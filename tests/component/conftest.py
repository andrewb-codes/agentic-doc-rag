from collections.abc import AsyncGenerator

import pytest
from httpx import AsyncClient

from tests.helpers import app_client


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    async with app_client() as client:
        yield client
