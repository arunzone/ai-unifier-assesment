from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.orm.query import Query

from ai_unifier_assesment.db.session import get_db_session
from ai_unifier_assesment.models.metrics import Metric


class MetricsRepository:
    """SQLAlchemy implementation of metrics repository."""

    def __init__(self, session: Annotated[Session, Depends(get_db_session)]):
        self._session = session

    def create(
        self,
        endpoint: str,
        session_id: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: float,
        metadata: Optional[dict] = None,
    ) -> Metric:
        metric = Metric(
            timestamp=datetime.utcnow(),
            endpoint=endpoint,
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            extra_data=metadata or {},
        )
        self._session.add(metric)
        self._session.flush()
        return metric

    def get_all(
        self,
        endpoint: Optional[str] = None,
        limit: int = 1000,
    ) -> list[Metric]:
        query: Query[Metric] = self._session.query(Metric)

        if endpoint:
            query = query.filter(Metric.endpoint == endpoint)

        return query.order_by(desc(Metric.timestamp)).limit(limit).all()

    def get_recent(self, hours: int = 24, endpoint: Optional[str] = None) -> list[Metric]:
        since = datetime.utcnow() - timedelta(hours=hours)
        query: Query[Metric] = self._session.query(Metric).filter(Metric.timestamp >= since)

        if endpoint:
            query = query.filter(Metric.endpoint == endpoint)

        return query.order_by(desc(Metric.timestamp)).all()
