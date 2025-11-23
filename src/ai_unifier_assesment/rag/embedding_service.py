from typing import Annotated

from fastapi import Depends
from langchain_community.embeddings import OllamaEmbeddings

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings


class EmbeddingService:
    def __init__(self, settings: Annotated[Settings, Depends(get_cached_settings)]):
        self._settings = settings

    def get_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=self._settings.ollama.embedding_model,
            base_url=self._settings.ollama.base_url,
        )
