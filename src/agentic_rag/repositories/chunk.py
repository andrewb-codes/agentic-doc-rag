from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import DocumentChunk
from agentic_rag.services.chunk import TextChunk


class DocumentChunkRepository:
    def __init__(self, *, session: AsyncSession) -> None:
        self.session = session

    async def create_chunks(
        self,
        *,
        document_id: int,
        chunks_list: list[TextChunk],
    ) -> list[DocumentChunk]:
        document_chunks = [
            DocumentChunk(
                document_id=document_id,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                source=chunk.source,
            )
            for chunk in chunks_list
        ]

        self.session.add_all(document_chunks)
        await self.session.flush()
        return document_chunks

    async def get_by_ids(self, *, chunk_ids: list[int]) -> list[DocumentChunk]:
        if not chunk_ids:
            return []

        query = select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        chunks = list(await self.session.scalars(query))
        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]
