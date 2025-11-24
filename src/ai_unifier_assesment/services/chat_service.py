import json
import time
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.stream_metrics import StreamMetrics, TokenCounter
from ai_unifier_assesment.services.memory_service import MemoryService


class ChatService:
    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        metrics: Annotated[StreamMetrics, Depends(StreamMetrics)],
        memory_service: Annotated[MemoryService, Depends(MemoryService)],
        token_counter: Annotated[TokenCounter, Depends(TokenCounter)],
    ):
        self._model = model
        self._metrics = metrics
        self._memory_service = memory_service
        self._token_counter = token_counter

    async def stream_response(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        start_time = time.time()
        completion_text = ""

        chain_with_history = await self._build_chain()

        # Build messages for token counting
        history = self._memory_service.get_session_history(session_id)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in history.messages:
            role = "assistant" if msg.type == "ai" else "user"
            messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        async for chunk in chain_with_history.astream(
            {"message": message},
            config={"configurable": {"session_id": session_id}},
        ):
            if chunk.content:
                completion_text += chunk.content
                yield f"data: {chunk.content}\n\n"

        # Count tokens using tiktoken
        prompt_tokens = self._token_counter.count_message_tokens(messages)
        completion_tokens = self._token_counter.count_text_tokens(completion_text)
        stats = self._metrics.build_stats(start_time, prompt_tokens, completion_tokens)
        yield f"event: stats\ndata: {json.dumps(stats)}\n\n"

    async def _build_chain(self) -> RunnableWithMessageHistory:
        llm = self._model.get_chat_model()
        trimmer = self._memory_service.get_trimmer()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{message}"),
            ]
        )

        chain = (
            {"chat_history": lambda x: trimmer.invoke(x["chat_history"]), "message": lambda x: x["message"]}
            | prompt
            | llm
        )

        return RunnableWithMessageHistory(
            chain,
            self._memory_service.get_session_history,
            input_messages_key="message",
            history_messages_key="chat_history",
        )
