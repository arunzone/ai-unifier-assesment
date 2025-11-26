import json
import time
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.services.stream_metrics import StreamMetrics, TokenCounter
from ai_unifier_assesment.services.memory_service import MemoryService
from ai_unifier_assesment.repositories.metrics_repository import MetricsRepository


class ChatService:
    def __init__(
        self,
        model: Annotated[Model, Depends(Model)],
        metrics: Annotated[StreamMetrics, Depends(StreamMetrics)],
        memory_service: Annotated[MemoryService, Depends(MemoryService)],
        token_counter: Annotated[TokenCounter, Depends(TokenCounter)],
        metrics_repo: Annotated[MetricsRepository, Depends(MetricsRepository)],
    ):
        self._model = model
        self._metrics = metrics
        self._memory_service = memory_service
        self._token_counter = token_counter
        self._metrics_repo = metrics_repo

    @staticmethod
    def _build_messages_for_token_counting(history_messages, user_message: str) -> list[dict]:
        """Build messages list for token counting."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for msg in history_messages.messages:
            role = "assistant" if msg.type == "ai" else "user"
            messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _persist_metrics(self, session_id: str, prompt_tokens: int, completion_tokens: int, stats: dict) -> None:
        """Persist metrics to database with error handling."""
        try:
            self._metrics_repo.create(
                endpoint="chat",
                session_id=session_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=stats["cost"],
                latency_ms=stats["latency_ms"],
            )
        except Exception as e:
            # Log error but don't fail the request
            import logging

            logging.getLogger(__name__).error(f"Failed to persist metrics: {e}")

    async def stream_response(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        start_time = time.time()
        completion_text = ""

        chain_with_history = await self._build_chain()

        # Build messages for token counting
        history = self._memory_service.get_session_history(session_id)
        messages = self._build_messages_for_token_counting(history, message)

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

        # Persist metrics to database
        self._persist_metrics(session_id, prompt_tokens, completion_tokens, stats)

        yield f"event: stats\ndata: {json.dumps(stats)}\n\n"

    async def _build_chain(self) -> RunnableWithMessageHistory:
        llm = self._model.stream_model()
        trimmer = self._memory_service.get_trimmer()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{message}"),
            ]
        )

        chain: Runnable[Any, Any] = (
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
