from typing import Annotated

from fastapi import Depends
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_settings


class MemoryService:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        self._settings = settings

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return PostgresChatMessageHistory(
            session_id=session_id,
            connection_string=self._settings.postgres.connection_string,
        )

    def get_trimmer(self):
        return trim_messages(
            strategy="last",
            max_tokens=self._settings.memory_window_size,
            token_counter=len,
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
