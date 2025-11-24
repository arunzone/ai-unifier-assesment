"""Code healing API endpoints."""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_unifier_assesment.agent.self_healing_agent import SelfHealingAgent

logger = logging.getLogger(__name__)

router = APIRouter()


class CodeHealingRequest(BaseModel):
    """Request to generate and heal code."""

    task_description: str = Field(
        description="Natural language description of the coding task",
        examples=["write quicksort in Rust", "implement binary search in Python"],
    )
    language: Literal["python", "rust"] = Field(description="Programming language to generate code in")


class CodeHealingResponse(BaseModel):
    """Response from code healing process."""

    success: bool = Field(description="Whether all tests passed")
    attempts: int = Field(description="Number of attempts made")
    final_code: str = Field(description="Final generated code")
    test_output: str = Field(description="Test execution output")
    working_directory: str = Field(description="Directory where code and tests were generated")
    message: str = Field(description="Final status message")


@router.post("/api/heal-code", response_model=CodeHealingResponse)
async def heal_code(
    request: CodeHealingRequest,
    agent: Annotated[SelfHealingAgent, Depends(SelfHealingAgent)],
) -> CodeHealingResponse:
    """Generate and self-heal code based on natural language description.

    This endpoint:
    1. Generates code from the task description
    2. Writes code to disk and runs tests
    3. Iteratively fixes errors until tests pass or max attempts reached
    4. Returns the final code and status

    Args:
        request: Contains task description and language
        agent: Self-healing agent instance

    Returns:
        CodeHealingResponse with final code and status

    Raises:
        HTTPException: If invalid language or processing error occurs
    """
    try:
        logger.info(f"Received code healing request: {request.task_description} ({request.language})")

        # Execute the self-healing loop
        final_state = await agent.heal(request.task_description, request.language)

        return CodeHealingResponse(
            success=final_state.success,
            attempts=final_state.attempt_number + 1,
            final_code=final_state.current_code or "",
            test_output=final_state.test_output or "",
            working_directory=final_state.working_directory,
            message=final_state.final_message,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error in heal_code: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e
