from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_rag.models import QAHistory, VerificationVerdict


class QAHistoryRepository:
    def __init__(self, *, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        user_id: int,
        document_id: int | None,
        question: str,
        answer: str,
        verification_verdict: VerificationVerdict = VerificationVerdict.NOT_VERIFIED,
    ) -> QAHistory:
        item = QAHistory(
            user_id=user_id,
            document_id=document_id,
            question=question,
            answer=answer,
            verification_verdict=verification_verdict,
        )
        self.session.add(item)
        await self.session.flush()
        return item

    async def list_by_user(self, *, user_id: int) -> list[QAHistory]:
        query = select(QAHistory).where(QAHistory.user_id == user_id).order_by(QAHistory.id.desc())
        return list(await self.session.scalars(query))
