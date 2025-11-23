#!/usr/bin/env python
"""
Generate synthetic evaluation dataset using Ragas framework.
Run this script AFTER ingestion to create evaluation questions.

Usage:
    python -m ai_unifier_assesment.generate_testset
    python -m ai_unifier_assesment.generate_testset --test-size 30
    python -m ai_unifier_assesment.generate_testset --stats
"""

import argparse
import logging
import sys
import time

from ai_unifier_assesment.config import get_settings
from ai_unifier_assesment.evaluation.evaluation_data_service import EvaluationDataService
from ai_unifier_assesment.evaluation.testset_generator_service import TestsetGeneratorService
from ai_unifier_assesment.rag.document_loader_service import DocumentLoaderService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_services():
    settings = get_settings()
    evaluation_service = EvaluationDataService(settings)
    generator_service = TestsetGeneratorService(settings)
    document_loader = DocumentLoaderService(
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
    )
    return settings, evaluation_service, generator_service, document_loader


def generate_testset(directory: str, test_size: int | None = None) -> None:
    settings, evaluation_service, generator_service, document_loader = create_services()

    evaluation_service.initialize_database()

    # Check if questions already exist
    existing_count = evaluation_service.get_question_count()
    if existing_count > 0:
        logger.info(f"Database already has {existing_count} evaluation questions, skipping generation")
        return

    logger.info(f"Loading documents from {directory}")
    documents = document_loader.load_and_split_directory(directory)
    logger.info(f"Loaded {len(documents)} document chunks")

    if not documents:
        logger.error("No documents found to generate testset")
        return

    start_time = time.time()
    questions = generator_service.generate(documents, test_size)
    elapsed = time.time() - start_time

    logger.info(f"Generated {len(questions)} questions in {elapsed:.2f}s")

    evaluation_service.save_questions_batch(questions)
    logger.info(f"Saved {len(questions)} questions to database")


def show_stats() -> None:
    settings, evaluation_service, _, _ = create_services()
    evaluation_service.initialize_database()

    count = evaluation_service.get_question_count()
    logger.info(f"Total evaluation questions: {count}")

    if count > 0:
        questions = evaluation_service.get_all_questions()
        logger.info("Sample questions:")
        for q in questions[:3]:
            logger.info(f"  - {q.question[:100]}...")


def clear_data() -> None:
    settings, evaluation_service, _, _ = create_services()
    evaluation_service.initialize_database()

    count = evaluation_service.clear_questions()
    logger.info(f"Cleared {count} questions from database")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation dataset using Ragas")
    parser.add_argument("--directory", type=str, default="data/corpus", help="Directory containing PDFs")
    parser.add_argument("--test-size", type=int, help="Number of test samples to generate")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--clear", action="store_true", help="Clear existing evaluation data")

    args = parser.parse_args()

    try:
        if args.stats:
            show_stats()
            return 0

        if args.clear:
            clear_data()
            return 0

        generate_testset(args.directory, args.test_size)
        return 0
    except Exception as e:
        logger.error(f"Failed to generate testset: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
