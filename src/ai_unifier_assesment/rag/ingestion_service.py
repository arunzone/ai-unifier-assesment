from typing import Annotated

from fastapi import Depends
from langchain_core.documents import Document

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.rag.document_loader_service import DocumentLoaderService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


class IngestionService:
    def __init__(
        self,
        settings: Annotated[Settings, Depends(get_cached_settings)],
        document_loader: Annotated[DocumentLoaderService, Depends(DocumentLoaderService)],
        vector_store_service: Annotated[VectorStoreService, Depends(VectorStoreService)],
    ):
        self._settings = settings
        self._document_loader = document_loader
        self._vector_store_service = vector_store_service

    def ingest_pdf(self, file_path: str, collection_name: str = "rag_corpus") -> int:
        chunks = self._document_loader.load_and_split(file_path)
        return self._store_chunks(chunks, collection_name)

    def ingest_directory(self, directory_path: str, collection_name: str = "rag_corpus") -> int:
        chunks = self._document_loader.load_and_split_directory(directory_path)
        return self._store_chunks(chunks, collection_name)

    def _store_chunks(self, chunks: list[Document], collection_name: str) -> int:
        if not chunks:
            return 0

        vector_store = self._vector_store_service.get_vector_store(collection_name)
        vector_store.add_documents(chunks)
        return len(chunks)

    def get_collection_stats(self, collection_name: str = "rag_corpus") -> dict:
        client = self._vector_store_service.get_client()
        try:
            collection = client.get_collection(collection_name)
            return {
                "collection_name": collection_name,
                "document_count": collection.count(),
            }
        except Exception:
            return {
                "collection_name": collection_name,
                "document_count": 0,
            }
