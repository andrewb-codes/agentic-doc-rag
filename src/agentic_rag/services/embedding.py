from typing import Protocol


class EmbeddingService(Protocol):
    @property
    def vector_size(self) -> int:
        pass

    async def embed_texts(self, *, texts: list[str]) -> list[list[float]]:
        pass


class FakeEmbeddingService:
    def __init__(self, *, vector_size: int = 3) -> None:
        self._vector_size = vector_size

    @property
    def vector_size(self) -> int:
        return self._vector_size

    async def embed_texts(self, *, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] * self._vector_size for index, _ in enumerate(texts)]
