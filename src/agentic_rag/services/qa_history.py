from agentic_rag.models import QAHistory
from agentic_rag.repositories.qa_history import QAHistoryRepository


class QAHistoryService:
    def __init__(self, *, repository: QAHistoryRepository) -> None:
        self.repository = repository

    async def list_user_qa_history(self, *, user_id: int) -> list[QAHistory]:
        return await self.repository.list_by_user(user_id=user_id)
