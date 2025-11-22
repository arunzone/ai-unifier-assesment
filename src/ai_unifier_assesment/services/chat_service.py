from typing import AsyncGenerator

from ai_unifier_assesment.large_language_model.model import Model
from langchain_core.prompts import ChatPromptTemplate
from fastapi import Depends
from typing import Annotated


class ChatService:
    def __init__(self, model: Annotated[Model, Depends(Model)]):
        self._model = model

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        llm = self._model.get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{message}"),
            ]
        )
        chain = prompt | llm

        async for chunk in chain.astream({"message": message}):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
