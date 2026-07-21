import tempfile
from pathlib import Path

from fastapi import UploadFile

from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import InvalidUploadError, UploadTooLargeError

ALLOWED_PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


def validate_pdf_upload(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise InvalidUploadError()

    if file.content_type not in ALLOWED_PDF_MIME_TYPES:
        raise InvalidUploadError()


async def save_upload_to_temp_file(file: UploadFile) -> Path:
    total_size = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = Path(temp_file.name)

        while chunk := await file.read(UPLOAD_CHUNK_SIZE_BYTES):
            total_size += len(chunk)

            if total_size > settings.max_upload_size_bytes:
                temp_path.unlink(missing_ok=True)
                raise UploadTooLargeError()

            temp_file.write(chunk)

    return temp_path
