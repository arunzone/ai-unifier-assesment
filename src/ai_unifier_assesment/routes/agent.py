from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ai_unifier_assesment.agent.trip_planner_agent import TripPlannerAgent

router = APIRouter()


class TripPlanRequest(BaseModel):
    prompt: str


class TripPlanResponse(BaseModel):
    itinerary: dict | None
    scratchpad: list[str]


@router.post("/api/plan-trip", response_model=TripPlanResponse)
async def plan_trip(
    request: TripPlanRequest,
    agent: Annotated[TripPlannerAgent, Depends(TripPlannerAgent)],
) -> TripPlanResponse:
    result = await agent.plan_trip(request.prompt)
    return TripPlanResponse(
        itinerary=result["itinerary"],
        scratchpad=result["scratchpad"],
    )
