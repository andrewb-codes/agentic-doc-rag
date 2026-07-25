from typing import Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from agentic_rag.models import DocumentChunk


class ChatCompletionsClient(Protocol):
    async def create(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int,
    ) -> ChatCompletion:
        pass


class OpenAIChatService:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        base_url: str | None = None,
        chat_completions_client: ChatCompletionsClient | None = None,
    ) -> None:
        self.chat_completions_client = (
            chat_completions_client
            if chat_completions_client is not None
            else AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=1,
                timeout=30.0,
            ).chat.completions
        )
        self.model = model
        self.max_tokens = max_tokens

    async def answer_question(self, *, question: str, chunks: list[DocumentChunk]) -> str:
        response = await self.chat_completions_client.create(
            model=self.model,
            messages=build_rag_messages(question=question, chunks=chunks),
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""


def build_rag_messages(
    *,
    question: str,
    chunks: list[DocumentChunk],
) -> list[ChatCompletionMessageParam]:
    context = "\n\n".join(
        f"[chunk_id={chunk.id}, document_id={chunk.document_id}, page={chunk.page}]\n{chunk.text}"
        for chunk in chunks
    )

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": (
            "Answer the user's question using only the provided document context. "
            "If the context does not contain the answer, say that the answer was not found "
            "in the documents."
        ),
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Question:\n{question}\n\nDocument context:\n{context}",
    }
    return [system_message, user_message]
