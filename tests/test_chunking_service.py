import pytest

from agentic_rag.services.chunk import InvalidChunkingConfigError, TextChunker
from agentic_rag.services.pdf import ExtractedPage

pytestmark = pytest.mark.no_db


def test_chunk_short_page_keeps_page_metadata() -> None:
    chunks = TextChunker(chunk_size=100, overlap=10).chunk_pages(
        pages=[ExtractedPage(page_number=3, text="Short text")],
        source="manual.pdf",
    )

    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Short text"
    assert chunks[0].source == "manual.pdf"


def test_chunk_long_text_with_overlap() -> None:
    chunks = TextChunker(chunk_size=10, overlap=3).chunk_pages(
        pages=[ExtractedPage(page_number=1, text="abcdefghijklmnopqrstuvwxyz")],
        source="manual.pdf",
    )

    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_chunk_index_increases_across_pages() -> None:
    chunks = TextChunker(chunk_size=5, overlap=0).chunk_pages(
        pages=[
            ExtractedPage(page_number=1, text="abcde"),
            ExtractedPage(page_number=2, text="fghij"),
        ],
        source="manual.pdf",
    )

    assert [(chunk.page, chunk.chunk_index, chunk.text, chunk.source) for chunk in chunks] == [
        (1, 0, "abcde", "manual.pdf"),
        (2, 1, "fghij", "manual.pdf"),
    ]


def test_empty_pages_do_not_create_chunks() -> None:
    chunks = TextChunker().chunk_pages(
        pages=[
            ExtractedPage(page_number=1, text=""),
            ExtractedPage(page_number=2, text="   "),
        ],
        source="manual.pdf",
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_config_raises_error(chunk_size: int, overlap: int) -> None:
    with pytest.raises(InvalidChunkingConfigError):
        TextChunker(chunk_size=chunk_size, overlap=overlap)
