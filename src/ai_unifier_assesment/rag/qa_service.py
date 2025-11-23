import time
from typing import Annotated

from fastapi import Depends
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


class QAService:
    def __init__(
        self,
        settings: Annotated[Settings, Depends(get_cached_settings)],
        vector_store_service: Annotated[VectorStoreService, Depends(VectorStoreService)],
    ):
        self._settings = settings
        self._vector_store_service = vector_store_service

    def get_llm(self) -> Ollama:
        return Ollama(
            model="llama3.2",
            base_url=self._settings.ollama.base_url,
        )

    def format_docs_with_citations(self, docs: list[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            formatted.append(f"[Source {i + 1}: {source}, Page {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """Answer the following question based ONLY on the provided context.
When you state a fact, place a citation next to it using the format [Source X] where X is the source number.

Context:
{context}

Question: {question}

Answer with citations:"""
        )

    def create_chain(self, retriever):
        prompt = self.get_prompt()
        llm = self.get_llm()

        chain = (
            {"context": retriever | self.format_docs_with_citations, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def answer(self, question: str, collection_name: str = "rag_corpus") -> dict:
        retriever = self._vector_store_service.get_retriever(collection_name)

        start_time = time.time()
        docs = retriever.invoke(question)
        retrieval_time_ms = (time.time() - start_time) * 1000

        chain = self.create_chain(retriever)
        answer = chain.invoke(question)

        return {
            "answer": answer,
            "sources": [
                {"source": doc.metadata.get("source", "Unknown"), "page": doc.metadata.get("page", "N/A")}
                for doc in docs
            ],
            "retrieval_time_ms": round(retrieval_time_ms, 2),
        }

    def retrieve_only(self, question: str, collection_name: str = "rag_corpus", k: int = 5) -> dict:
        retriever = self._vector_store_service.get_retriever(collection_name, k)

        start_time = time.time()
        docs = retriever.invoke(question)
        retrieval_time_ms = (time.time() - start_time) * 1000

        return {
            "documents": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                }
                for doc in docs
            ],
            "retrieval_time_ms": round(retrieval_time_ms, 2),
        }
