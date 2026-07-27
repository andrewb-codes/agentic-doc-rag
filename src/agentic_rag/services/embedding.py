from typing import Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from agentic_rag.core.exceptions import EmbeddingProviderError, EmbeddingProviderTimeoutError


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


class OpenAIEmbeddingService:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        vector_size: int = 1536,
        client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 0,
    ) -> None:
        self.client = (
            client
            if client is not None
            else AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=max_retries,
                timeout=timeout_seconds,
            )
        )
        self.model = model
        self._vector_size = vector_size

    @property
    def vector_size(self) -> int:
        return self._vector_size

    async def embed_texts(self, *, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
        except APITimeoutError as exc:
            raise EmbeddingProviderTimeoutError() from exc
        except (APIConnectionError, APIStatusError) as exc:
            raise EmbeddingProviderError() from exc

        return [item.embedding for item in response.data]
