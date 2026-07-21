from pathlib import Path

import fitz
import pytest

from agentic_rag.services.pdf import EmptyPdfError, InvalidPdfError, PdfExtractor

pytestmark = pytest.mark.no_db


def create_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()

    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)

    document.save(path)
    document.close()


def test_extract_pdf_text_with_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path, ["First page", "Second page"])

    extracted = PdfExtractor().extract(path=pdf_path)

    assert extracted.page_count == 2
    assert [page.page_number for page in extracted.pages] == [1, 2]
    assert [page.text for page in extracted.pages] == ["First page", "Second page"]
    assert extracted.text == "First page\n\nSecond page"


def test_extract_empty_text_pdf_raises_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    create_pdf(pdf_path, [""])

    with pytest.raises(EmptyPdfError, match="no extractable text"):
        PdfExtractor().extract(path=pdf_path)


def test_extract_invalid_pdf_raises_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf")

    with pytest.raises(InvalidPdfError, match="Could not read PDF file"):
        PdfExtractor().extract(path=pdf_path)
