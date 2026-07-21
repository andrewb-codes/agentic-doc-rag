from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedPdf:
    page_count: int
    pages: list[ExtractedPage]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class PdfExtractionError(Exception):
    pass


class EmptyPdfError(PdfExtractionError):
    pass


class InvalidPdfError(PdfExtractionError):
    pass


class PdfExtractor:
    def extract(self, *, path: Path) -> ExtractedPdf:
        try:
            with fitz.open(path) as document:
                page_count = document.page_count
                pages = [
                    ExtractedPage(
                        page_number=page_index + 1,
                        text=document.load_page(page_index).get_text().strip(),
                    )
                    for page_index in range(page_count)
                ]
        except Exception as exc:
            raise InvalidPdfError("Could not read PDF file") from exc

        if page_count == 0:
            raise EmptyPdfError("PDF has no pages")

        if not any(page.text for page in pages):
            raise EmptyPdfError("PDF has no extractable text")

        return ExtractedPdf(page_count=page_count, pages=pages)
