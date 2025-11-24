from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DECIMAL, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ai_unifier_assesment.models.base import Base


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    endpoint: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost: Mapped[float] = mapped_column(DECIMAL(10, 8), nullable=False)
    latency_ms: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=False)
    extra_data: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
