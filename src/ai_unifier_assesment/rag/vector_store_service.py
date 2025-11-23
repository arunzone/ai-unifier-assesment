from typing import Annotated
import logging

import chromadb
from chromadb import ClientAPI
from fastapi import Depends
from langchain_chroma import Chroma

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.rag.embedding_service import EmbeddingService
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class VectorStoreService:
    def __init__(
        self,
        settings: Annotated[Settings, Depends(get_cached_settings)],
        embedding_service: Annotated[EmbeddingService, Depends(EmbeddingService)],
    ):
        self._settings = settings
        self._embedding_service = embedding_service
        self._logger = logging.getLogger(__name__)

    def get_client(self) -> ClientAPI:
        try:
            return chromadb.HttpClient(
                host=self._settings.chroma.host,
                port=self._settings.chroma.port,
            )
        except Exception as e:
            self._logger.error(
                f"Failed to connect to ChromaDB at {self._settings.chroma.host}:{self._settings.chroma.port}: {e}"
            )
            raise ConnectionError(f"Unable to establish connection to ChromaDB server: {e}") from e

    def get_vector_store(self, collection_name: str = "rag_corpus") -> VectorStore:
        return Chroma(
            client=self.get_client(),
            collection_name=collection_name,
            embedding_function=self._embedding_service.get_embeddings(),
        )

    def get_retriever(
        self,
        collection_name: str = "rag_corpus",
        k: int = 5,
        search_type: str = "mmr",
        **kwargs,
    ) -> BaseRetriever:
        vector_store = self.get_vector_store(collection_name)
        search_kwargs = {"k": k, "fetch_k": 20}
        search_kwargs.update(kwargs)

        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
