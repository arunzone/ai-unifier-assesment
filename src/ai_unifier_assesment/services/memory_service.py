from typing import Annotated, Dict

from fastapi import Depends
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_settings


class MemoryService:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        self._store: Dict[str, InMemoryChatMessageHistory] = {}
        self._settings = settings

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def get_trimmer(self):
        return trim_messages(
            strategy="last",
            max_tokens=self._settings.memory_window_size,
            token_counter=len,
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
