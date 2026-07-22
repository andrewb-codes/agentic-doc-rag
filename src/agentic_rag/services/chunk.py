from dataclasses import dataclass

from agentic_rag.services.pdf import ExtractedPage


@dataclass(frozen=True)
class TextChunk:
    page: int
    chunk_index: int
    text: str
    source: str


class ChunkingError(Exception):
    pass


class InvalidChunkingConfigError(ChunkingError):
    pass


class TextChunker:
    def __init__(self, *, chunk_size: int = 1200, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise InvalidChunkingConfigError("chunk_size must be positive")

        if overlap < 0:
            raise InvalidChunkingConfigError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise InvalidChunkingConfigError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, *, pages: list[ExtractedPage], source: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            for text in self._chunk_text(page.text):
                chunks.append(
                    TextChunk(
                        page=page.page_number,
                        chunk_index=chunk_index,
                        text=text,
                        source=source,
                    )
                )
                chunk_index += 1

        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        normalized_text = " ".join(text.split())

        if not normalized_text:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(normalized_text):
            end = min(start + self.chunk_size, len(normalized_text))
            chunk = normalized_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(normalized_text):
                break

            start = end - self.overlap

        return chunks
