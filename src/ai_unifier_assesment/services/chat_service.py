import json
import time
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.stream_metrics import StreamMetrics
from ai_unifier_assesment.services.memory_service import MemoryService


class ChatService:
    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        metrics: Annotated[StreamMetrics, Depends(StreamMetrics)],
        memory_service: Annotated[MemoryService, Depends(MemoryService)],  # Injected
    ):
        self._model = model
        self._metrics = metrics
        self._memory_service = memory_service

    async def stream_response(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0

        llm = self._model.get_chat_model()
        trimmer = self._memory_service.get_trimmer()
        prompt = await self.prompt_templated()
        chain_with_history = await self.chain_from(llm, prompt, trimmer)

        async for chunk in chain_with_history.astream(
            {"message": message}, config={"configurable": {"session_id": session_id}}
        ):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"

            prompt_tokens, completion_tokens = self._metrics.extract_tokens(chunk)
        stats = self._metrics.build_stats(start_time, prompt_tokens, completion_tokens)

        yield f"event: stats\ndata: {json.dumps(stats)}\n\n"

    async def chain_from(self, llm, prompt, trimmer) -> RunnableWithMessageHistory:
        chain = (
            {"chat_history": lambda x: trimmer.invoke(x["chat_history"]), "message": lambda x: x["message"]}
            | prompt
            | llm
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,
            self._memory_service.get_session_history,  # Pass the bound method
            input_messages_key="message",
            history_messages_key="chat_history",
        )
        return chain_with_history

    async def prompt_templated(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{message}"),
            ]
        )
