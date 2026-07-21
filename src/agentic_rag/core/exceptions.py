class AppError(Exception):
    status_code = 500
    detail = "error.internal"


class UnauthorizedError(AppError):
    status_code = 401
    detail = "error.auth.unauthorized"


class DocumentNotFoundError(AppError):
    status_code = 404
    detail = "error.document.not_found"


class InvalidUploadError(AppError):
    status_code = 400
    detail = "error.upload.invalid"


class UploadTooLargeError(AppError):
    status_code = 413
    detail = "error.upload.too_large"


class PdfProcessingError(AppError):
    status_code = 422
    detail = "error.pdf.processing_failed"
