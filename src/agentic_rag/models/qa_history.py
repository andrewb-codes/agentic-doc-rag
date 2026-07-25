from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_rag.db.base import Base

if TYPE_CHECKING:
    from agentic_rag.models.document import Document
    from agentic_rag.models.user import User


class VerificationVerdict(enum.StrEnum):
    NOT_VERIFIED = "not_verified"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class QAHistory(Base):
    __tablename__ = "qa_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    verification_verdict: Mapped[VerificationVerdict] = mapped_column(
        Enum(
            VerificationVerdict,
            name="verification_verdict",
            values_callable=lambda enum_class: [item.value for item in enum_class],
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    user: Mapped[User] = relationship(back_populates="qa_history")
    document: Mapped[Document | None] = relationship(back_populates="qa_history")
