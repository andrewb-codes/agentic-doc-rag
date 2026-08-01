import enum


class AnswerStatus(enum.StrEnum):
    ANSWERED = "answered"
    NOT_FOUND = "not_found"


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class VerificationVerdict(enum.StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
