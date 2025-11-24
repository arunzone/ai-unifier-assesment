from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ai_unifier_assesment.repositories.metrics_repository import MetricsRepository

router = APIRouter()


class MetricResponse(BaseModel):
    id: int
    timestamp: str
    endpoint: str
    session_id: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    metadata: Optional[dict]

    class Config:
        from_attributes = True


@router.get("/api/metrics", response_model=list[MetricResponse])
async def get_metrics(
    endpoint: Optional[str] = Query(None, description="Filter by endpoint (chat or agent)"),
    hours: int = Query(24, description="Get metrics from last N hours"),
    metrics_repo: Annotated[MetricsRepository, Depends(MetricsRepository)] = None,
):
    """Retrieve metrics for dashboard visualization."""
    metrics = metrics_repo.get_recent(hours=hours, endpoint=endpoint)

    return [
        MetricResponse(
            id=m.id,
            timestamp=m.timestamp.isoformat(),
            endpoint=m.endpoint,
            session_id=m.session_id,
            prompt_tokens=m.prompt_tokens,
            completion_tokens=m.completion_tokens,
            total_tokens=m.total_tokens,
            cost=float(m.cost),
            latency_ms=float(m.latency_ms),
            metadata=m.extra_data,
        )
        for m in metrics
    ]
