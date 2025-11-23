import logging
from typing import Annotated, cast

from fastapi import Depends
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import embedding_factory
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator, Testset
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms,
    default_transforms,
)

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.large_language_model.model import Model


class TestsetGeneratorService:
    __test__ = False

    def __init__(self, settings: Annotated[Settings, Depends(get_cached_settings)]):
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        self._model = Model(settings)

    def get_raw_llm(self):
        return ChatOllama(
            model=self._settings.evaluation_llm_model, base_url=self._settings.ollama.base_url, format="json"
        )

    def get_raw_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=self._settings.ollama.embedding_model,
            base_url=self._settings.ollama.base_url,
        )

    def build_knowledge_graph(
        self, documents: list[Document], llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings
    ) -> KnowledgeGraph:
        kg = KnowledgeGraph()

        for doc in documents:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )

        transforms = default_transforms(documents=documents, llm=llm, embedding_model=embeddings)
        apply_transforms(kg, transforms)

        return kg

    def generate(self, documents: list[Document], test_size: int | None = None) -> list[dict]:
        if test_size is None:
            test_size = self._settings.evaluation.test_size

        self._logger.info(f"Generating {test_size} test samples from {len(documents)} documents")

        raw_llm = self.get_raw_llm()
        raw_embeddings = self.get_raw_embeddings()

        ragas_llm = LangchainLLMWrapper(raw_llm)
        ragas_embeddings = embedding_factory(raw_embeddings)

        self._logger.info("Building knowledge graph...")
        kg = self.build_knowledge_graph(documents, ragas_llm, ragas_embeddings)

        generator = TestsetGenerator(
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
            knowledge_graph=kg,
        )

        result = generator.generate(testset_size=test_size)
        testset = cast(Testset, result)

        df = testset.to_pandas()
        results = []

        for _, row in df.iterrows():
            item = {
                "question": row.get("user_input", row.get("question", "")),
                "ground_truth_answer": row.get("reference", row.get("ground_truth", "")),
                "ground_truth_contexts": row.get("reference_contexts", row.get("contexts", [])),
                "source_metadata": {
                    "synthesizer_name": row.get("synthesizer_name", "unknown"),
                },
            }
            results.append(item)

        self._logger.info(f"Generated {len(results)} evaluation samples")
        return results
