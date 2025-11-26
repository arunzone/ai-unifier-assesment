from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from ai_unifier_assesment.config import Settings, get_settings
from fastapi import Depends
from typing import Annotated
from pydantic import SecretStr


class Model:
    def __init__(self, settings: Annotated[Settings, Depends(get_settings)]):
        self._settings = settings

    def stream_model(self) -> BaseChatModel:
        return ChatOpenAI(
            base_url=self._settings.openai.base_url,
            api_key=SecretStr(self._settings.openai.api_key),
            model=self._settings.openai.model_name,
            streaming=True,
            stream_usage=True,
            model_kwargs={"stream_options": {"include_usage": True}},
        )

    def get_chat_model_for_evaluation(self) -> BaseChatModel:
        return ChatOpenAI(
            base_url=self._settings.openai.base_url,
            api_key=SecretStr(self._settings.openai.api_key),
            model=self._settings.openai.model_name,
            streaming=False,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def simple_model(self) -> BaseChatModel:
        return ChatOpenAI(
            base_url=self._settings.openai.base_url,
            api_key=SecretStr(self._settings.openai.api_key),
            model=self._settings.openai.model_name,
            streaming=False,
        )
