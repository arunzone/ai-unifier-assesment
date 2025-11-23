import logging
import time
from typing import Any, Annotated, Dict

from fastapi import Depends

from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.dependencies import get_cached_settings
from ai_unifier_assesment.evaluation.evaluation_data_service import EvaluationDataService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService


class BenchmarkService:
    def __init__(
        self,
        settings: Annotated[Settings, Depends(get_cached_settings)],
        evaluation_service: Annotated[EvaluationDataService, Depends(EvaluationDataService)],
        vector_store_service: Annotated[VectorStoreService, Depends(VectorStoreService)],
    ):
        self._settings = settings
        self._evaluation_service = evaluation_service
        self._vector_store_service = vector_store_service
        self._logger = logging.getLogger(__name__)

    def run_retrieval_benchmark(self, k: int = 5) -> Dict[str, Any]:
        questions = self._evaluation_service.get_all_questions()

        if not questions:
            return {"error": "No evaluation questions found", "total_questions": 0}

        retriever = self._vector_store_service.get_retriever(self._settings.chroma.collection_name, k=k)
        results, retrieval_times, hits = self._evaluate_questions(questions, retriever)

        return self._build_benchmark_result(k, len(questions), hits, retrieval_times, results)

    def _evaluate_questions(self, questions: list, retriever) -> tuple[list, list[float], int]:
        retrieval_times: list[float] = []
        hits = 0
        results = []

        for q in questions:
            result = self._evaluate_single_question(q, retriever)
            results.append(result)
            retrieval_times.append(result["retrieval_time_ms"])
            if result["hit"]:
                hits += 1

        return results, retrieval_times, hits

    def _evaluate_single_question(self, question, retriever) -> dict:
        start_time = time.time()
        retrieved_docs = retriever.invoke(question.question)
        retrieval_time_ms = (time.time() - start_time) * 1000

        retrieved_contents = [doc.page_content for doc in retrieved_docs]
        hit = self._check_hit(question.ground_truth_contexts, retrieved_contents)

        return {
            "question_id": question.id,
            "question": question.question[:100],
            "hit": hit,
            "retrieval_time_ms": round(retrieval_time_ms, 2),
        }

    def _build_benchmark_result(
        self, k: int, total: int, hits: int, times: list[float], details: list
    ) -> Dict[str, Any]:
        median_time = self._calculate_median(times)
        avg_time = sum(times) / len(times) if times else 0

        return {
            "total_questions": total,
            "top_k": k,
            "hits": hits,
            "accuracy_percent": round((hits / total) * 100, 2) if total else 0,
            "median_retrieval_time_ms": round(median_time, 2),
            "avg_retrieval_time_ms": round(avg_time, 2),
            "min_retrieval_time_ms": round(min(times), 2) if times else 0,
            "max_retrieval_time_ms": round(max(times), 2) if times else 0,
            "meets_latency_requirement": median_time <= 300,
            "details": details,
        }

    def _check_hit(self, ground_truth_contexts: list[str], retrieved_contents: list[str]) -> bool:
        for gt_context in ground_truth_contexts:
            if self._matches_any_retrieved(gt_context.lower().strip(), retrieved_contents):
                return True
        return False

    def _matches_any_retrieved(self, gt_normalized: str, retrieved_contents: list[str]) -> bool:
        for retrieved in retrieved_contents:
            if self._is_match(gt_normalized, retrieved.lower().strip()):
                return True
        return False

    def _is_match(self, gt_normalized: str, retrieved_normalized: str) -> bool:
        if gt_normalized in retrieved_normalized or retrieved_normalized in gt_normalized:
            return True
        return self._calculate_overlap(gt_normalized, retrieved_normalized) > 0.5

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def _calculate_median(self, values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]
