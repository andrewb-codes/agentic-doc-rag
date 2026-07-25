from pathlib import Path

import fitz


def create_pdf(path: Path, text: str = "PDF text") -> None:
    document = fitz.open()
    page = document.new_page()

    if text:
        page.insert_text((72, 72), text)

    document.save(path)
    document.close()
