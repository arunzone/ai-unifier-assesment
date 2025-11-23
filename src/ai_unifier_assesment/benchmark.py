#!/usr/bin/env python
"""
Benchmark script for RAG retrieval accuracy and latency.
Reports top-5 retrieval accuracy on evaluation questions.

Usage:
    python -m ai_unifier_assesment.benchmark
    python -m ai_unifier_assesment.benchmark --k 10
    python -m ai_unifier_assesment.benchmark --verbose
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict

from ai_unifier_assesment.config import get_settings
from ai_unifier_assesment.evaluation.benchmark_service import BenchmarkService
from ai_unifier_assesment.evaluation.evaluation_data_service import EvaluationDataService
from ai_unifier_assesment.rag.embedding_service import EmbeddingService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_benchmark_service() -> BenchmarkService:
    settings = get_settings()
    evaluation_service = EvaluationDataService(settings)
    embedding_service = EmbeddingService(settings)
    vector_store_service = VectorStoreService(settings, embedding_service)
    return BenchmarkService(settings, evaluation_service, vector_store_service)


def run_benchmark(k: int = 5, verbose: bool = False, save: bool = False) -> Dict[str, Any]:
    logger.info(f"Running top-{k} retrieval benchmark...")

    service = create_benchmark_service()
    results: Dict[str, Any] = service.run_retrieval_benchmark(k=k)

    if "error" in results:
        logger.error(results["error"])
        return results

    _print_summary_report(results, k)
    if verbose:
        _print_detailed_results(results)

    if save:
        run_id = service.save_benchmark_result(results)
        print(f"Results saved to database with run_id: {run_id}")

    return results


def _print_summary_report(results: Dict[str, Any], k: int) -> None:
    print("\n" + "=" * 60)
    print("RAG RETRIEVAL BENCHMARK REPORT")
    print("=" * 60)
    print(f"Total Questions: {results['total_questions']}")
    print(
        f"Top-{k} Retrieval Accuracy: {results['accuracy_percent']}% "
        f"({results['hits']}/{results['total_questions']} hits)"
    )
    print(f"Median Retrieval Time: {results['median_retrieval_time_ms']} ms")
    print(f"Average Retrieval Time: {results['avg_retrieval_time_ms']} ms")
    print(f"Min Retrieval Time: {results['min_retrieval_time_ms']} ms")
    print(f"Max Retrieval Time: {results['max_retrieval_time_ms']} ms")
    print("-" * 60)

    if results["meets_latency_requirement"]:
        print("✓ PASS: Meets ≤300ms median retrieval time requirement")
    else:
        print("✗ FAIL: Does NOT meet ≤300ms median retrieval time requirement")

    print("=" * 60 + "\n")


def _print_detailed_results(results: Dict[str, Any]) -> None:
    print("Detailed Results:")
    print("-" * 60)
    for detail in results["details"]:
        status = "✓" if detail["hit"] else "✗"
        print(f"{status} Q{detail['question_id']}: {detail['question']}...")
        print(f"   Retrieval time: {detail['retrieval_time_ms']} ms")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG retrieval benchmark")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-question results")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--save", action="store_true", help="Save results to PostgreSQL database")

    args = parser.parse_args()

    try:
        results = run_benchmark(k=args.k, verbose=args.verbose, save=args.save)

        if args.json:
            print(json.dumps(results, indent=2))

        if "error" in results:
            return 1

        # Return non-zero if latency requirement not met
        if not results.get("meets_latency_requirement", False):
            return 1

        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
