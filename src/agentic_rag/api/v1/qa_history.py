from typing import Annotated

from fastapi import APIRouter, Depends

from agentic_rag.api.deps import get_current_telegram_user, get_qa_history_service
from agentic_rag.api.presenters.qa_history import build_qa_history_response
from agentic_rag.models import User
from agentic_rag.schemas.qa_history import QAHistoryResponse
from agentic_rag.services.qa_history import QAHistoryService

router = APIRouter(
    prefix="/qa-history",
    tags=["QA History"],
)


@router.get("", response_model=list[QAHistoryResponse])
async def list_user_qa_history(
    current_user: Annotated[User, Depends(get_current_telegram_user)],
    service: Annotated[QAHistoryService, Depends(get_qa_history_service)],
) -> list[QAHistoryResponse]:
    items = await service.list_user_qa_history(user_id=current_user.id)
    return [build_qa_history_response(item) for item in items]
