from enum import StrEnum


class RateLimitScope(StrEnum):
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_SEARCH = "document_search"
    DOCUMENT_ASK = "document_ask"
    QA_HISTORY_READ = "qa_history_read"
