import logging
from typing import Annotated, cast

from fastapi import Depends
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator, Testset
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import (
    apply_transforms,
    default_transforms,
)
from ragas.embeddings.base import BaseRagasEmbeddings  # Ragas Embeddings Base

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

    # FIX: Use the Ragas-wrapped models for the graph build
    def build_knowledge_graph(self, documents: list[Document], llm, embeddings: BaseRagasEmbeddings) -> KnowledgeGraph:
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

        # Apply default transforms to enrich the knowledge graph
        # NOTE: This function expects the *raw* LLM/Embeddings
        # OR the graph should be built externally.
        # Using Ragas-wrapped models for 'llm' and 'embedding_model' here is safer.
        transforms = default_transforms(documents=documents, llm=llm, embedding_model=embeddings)
        apply_transforms(kg, transforms)

        return kg

    def generate(self, documents: list[Document], test_size: int | None = None) -> list[dict]:
        if test_size is None:
            test_size = self._settings.evaluation.test_size

        self._logger.info(f"Generating {test_size} test samples from {len(documents)} documents")

        # STEP 1: Get RAW models
        raw_llm = self.get_raw_llm()
        raw_embeddings = self.get_raw_embeddings()

        # STEP 2: Wrap models for Ragas consistency
        ragas_llm = LangchainLLMWrapper(raw_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

        # STEP 3: Build Knowledge Graph
        self._logger.info("Building knowledge graph...")
        # FIX: Pass the Ragas-wrapped models to the graph builder
        kg = self.build_knowledge_graph(documents, ragas_llm, ragas_embeddings)

        # STEP 4: Initialize TestsetGenerator
        generator = TestsetGenerator(
            # FIX: Use the Ragas-wrapped models
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
            knowledge_graph=kg,
            # Note: The new graph-based method often omits the old 'evolutions'
            # and uses internal synthesizers based on the KG structure.
        )

        # Use default query distribution
        result = generator.generate(testset_size=test_size)
        testset = cast(Testset, result)

        df = testset.to_pandas()
        results = []

        for _, row in df.iterrows():
            item = {
                # Standardize column access for newer Ragas
                "question": row.get("user_input", row.get("question", "")),
                "ground_truth_answer": row.get("reference", row.get("ground_truth", "")),
                "ground_truth_contexts": row.get("reference_contexts", row.get("contexts", [])),
                "source_metadata": {
                    "synthesizer_name": row.get("synthesizer_name", "unknown"),
                    # Note: You can retrieve more metadata from the KG nodes if needed
                },
            }
            results.append(item)

        self._logger.info(f"Generated {len(results)} evaluation samples")
        return results
