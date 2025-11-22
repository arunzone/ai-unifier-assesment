from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from ai_unifier_assesment.config import Settings, get_settings
from fastapi import Depends
from typing import Annotated


class Model:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        self._settings = settings

    def get_chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            base_url=self._settings.openai.base_url,
            api_key=self._settings.openai.api_key,
            model=self._settings.openai.model_name,
            streaming=True,
            stream_usage=True,
        )
