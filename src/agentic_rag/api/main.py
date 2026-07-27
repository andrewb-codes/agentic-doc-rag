from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.api.v1.documents import router as documents_router
from agentic_rag.api.v1.qa_history import router as qa_history_router
from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import AppError
from agentic_rag.core.logging import configure_logging
from agentic_rag.middleware.request_logging import request_logging_middleware
from agentic_rag.rate_limit.service import RateLimitService

configure_logging(settings)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("application_starting")

    rate_limiter = RateLimitService.from_settings(settings)

    if rate_limiter.enabled and not await rate_limiter.check_storage():
        logger.error("rate_limit_storage_unavailable")
        raise RuntimeError("Rate limit Redis storage is not available")

    app.state.rate_limiter = rate_limiter
    logger.info("application_started")

    try:
        yield
    finally:
        logger.info("application_stopping")
        app.state.rate_limiter = None
        logger.info("application_stopped")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend API for agentic document RAG.",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.middleware("http")(request_logging_middleware)

app.include_router(documents_router)
app.include_router(qa_history_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    headers = None

    if hasattr(exc, "retry_after"):
        headers = {"Retry-After": str(exc.retry_after)}

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers,
    )
