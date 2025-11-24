import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai_unifier_assesment.agent.trip_planner_agent import TripPlannerAgent
from ai_unifier_assesment.repositories.metrics_repository import MetricsRepository

router = APIRouter()
logger = logging.getLogger(__name__)


class TripPlanRequest(BaseModel):
    prompt: str


class TripPlanResponse(BaseModel):
    itinerary: dict | None


@router.post("/api/plan-trip", response_model=TripPlanResponse)
async def plan_trip(
    request: TripPlanRequest,
    agent: Annotated[TripPlannerAgent, Depends(TripPlannerAgent)],
    metrics_repo: Annotated[MetricsRepository, Depends(MetricsRepository)],
) -> TripPlanResponse:
    start_time = time.time()
    result = await agent.plan_trip(request.prompt)
    latency_ms = (time.time() - start_time) * 1000

    # Persist metrics (agent doesn't return token counts yet, so we'll estimate or skip)
    try:
        metrics_repo.create(
            endpoint="agent",
            session_id=None,
            prompt_tokens=0,  # TODO: Implement token counting for agent
            completion_tokens=0,
            cost=0.0,  # TODO: Calculate cost based on model usage
            latency_ms=latency_ms,
            metadata={"prompt": request.prompt[:100]},  # Store truncated prompt
        )
    except Exception as e:
        logger.error(f"Failed to persist agent metrics: {e}")

    return TripPlanResponse(itinerary=result["itinerary"])
