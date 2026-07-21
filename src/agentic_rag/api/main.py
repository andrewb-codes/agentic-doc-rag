from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.api.v1.documents import router as documents_router
from agentic_rag.core.config import settings
from agentic_rag.core.exceptions import AppError
from agentic_rag.core.logging import configure_logging
from agentic_rag.middleware.request_logging import request_logging_middleware

configure_logging(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    headers = None

    if hasattr(exc, "retry_after"):
        headers = {"Retry-After": str(exc.retry_after)}

    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=headers
    )
