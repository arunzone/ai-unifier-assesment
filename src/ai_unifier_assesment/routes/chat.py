from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Annotated

from ai_unifier_assesment.services.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(ChatService)],
):
    return StreamingResponse(
        chat_service.stream_response(request.message),
        media_type="text/event-stream",
    )
