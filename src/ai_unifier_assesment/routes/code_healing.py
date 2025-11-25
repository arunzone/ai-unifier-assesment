"""Code healing API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_unifier_assesment.agent.self_healing_agent import CodingAgent

logger = logging.getLogger(__name__)

router = APIRouter()


class CodeHealingRequest(BaseModel):
    task_description: str = Field(
        description="Natural language description of the coding task. Language will be auto-detected.",
        examples=["write quicksort in Rust", "implement binary search in Python"],
    )


@router.post("/api/heal-code/stream")
async def heal_code_stream(
    request: CodeHealingRequest,
    agent: Annotated[CodingAgent, Depends(CodingAgent)],
):
    return StreamingResponse(
        agent.code_stream(request.task_description),
        media_type="text/event-stream",
    )
