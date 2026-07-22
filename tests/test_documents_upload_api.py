from pathlib import Path

import fitz
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from agentic_rag.db.session import AsyncSessionLocal
from agentic_rag.models import DocumentChunk
from tests.test_documents_api import internal_headers


def create_pdf(path: Path, text: str = "PDF text") -> None:
    document = fitz.open()
    page = document.new_page()

    if text:
        page.insert_text((72, 72), text)

    document.save(path)
    document.close()


async def test_upload_pdf_processes_document(client: AsyncClient, tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path)

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("manual.pdf", file, "application/pdf")},
        )

    body = response.json()

    assert response.status_code == 201
    assert body["filename"] == "manual.pdf"
    assert body["status"] == "processed"
    assert body["page_count"] == 1
    assert body["chunk_count"] == 1


async def test_upload_pdf_persists_document_chunks(client: AsyncClient, tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path)

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("manual.pdf", file, "application/pdf")},
        )

    document_id = response.json()["id"]

    async with AsyncSessionLocal() as session:
        chunks = list(
            await session.scalars(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
        )

    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "PDF text"
    assert chunks[0].source == "manual.pdf"


async def test_upload_rejects_non_pdf_extension(client: AsyncClient, tmp_path: Path) -> None:
    text_path = tmp_path / "manual.txt"
    text_path.write_text("not a pdf")

    with text_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("manual.txt", file, "application/pdf")},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "error.upload.invalid"}


async def test_upload_rejects_non_pdf_content_type(client: AsyncClient, tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path)

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("manual.pdf", file, "text/plain")},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "error.upload.invalid"}


async def test_upload_invalid_pdf_returns_processing_error(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf")

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("broken.pdf", file, "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "error.pdf.processing_failed"}


async def test_upload_rejects_too_large_file(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("agentic_rag.api.upload.settings.max_upload_size_bytes", 4)

    pdf_path = tmp_path / "manual.pdf"
    create_pdf(pdf_path)

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("manual.pdf", file, "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "error.upload.too_large"}


async def test_upload_invalid_pdf_creates_failed_document(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a pdf")

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("broken.pdf", file, "application/pdf")},
        )

    assert response.status_code == 422

    list_response = await client.get("/documents", headers=internal_headers())
    documents = list_response.json()

    assert list_response.status_code == 200
    assert len(documents) == 1
    assert documents[0]["filename"] == "broken.pdf"
    assert documents[0]["status"] == "failed"
    assert documents[0]["page_count"] is None
    assert documents[0]["chunk_count"] is None


async def test_upload_empty_text_pdf_returns_processing_error(
    client: AsyncClient,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "empty.pdf"
    create_pdf(pdf_path, text="")

    with pdf_path.open("rb") as file:
        response = await client.post(
            "/documents/upload",
            headers=internal_headers(),
            files={"file": ("empty.pdf", file, "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "error.pdf.processing_failed"}
