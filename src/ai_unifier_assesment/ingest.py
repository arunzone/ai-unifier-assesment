#!/usr/bin/env python
"""
Offline ingestion script for RAG system.
Run this script BEFORE starting the server to index documents.

Usage:
    python -m ai_unifier_assesment.ingest --pdf path/to/file.pdf
    python -m ai_unifier_assesment.ingest --directory path/to/pdfs/
    python -m ai_unifier_assesment.ingest --stats
"""

import argparse
import logging
import sys
import time

from ai_unifier_assesment.config import get_settings
from ai_unifier_assesment.rag.document_loader_service import DocumentLoaderService
from ai_unifier_assesment.rag.embedding_service import EmbeddingService
from ai_unifier_assesment.rag.ingestion_service import IngestionService
from ai_unifier_assesment.rag.vector_store_service import VectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_ingestion_service() -> IngestionService:
    settings = get_settings()
    embedding_service = EmbeddingService(settings)
    vector_store_service = VectorStoreService(settings, embedding_service)
    document_loader = DocumentLoaderService(
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
    )
    return IngestionService(settings, document_loader, vector_store_service)


def ingest_pdf(file_path: str) -> None:
    settings = get_settings()
    logger.info(f"Ingesting PDF: {file_path}")
    logger.info(f"Chunk size: {settings.rag.chunk_size}, Overlap: {settings.rag.chunk_overlap}")

    service = create_ingestion_service()
    collection_name = settings.chroma.collection_name

    start_time = time.time()
    chunk_count = service.ingest_pdf(file_path, collection_name)
    elapsed = time.time() - start_time

    logger.info(f"Ingested {chunk_count} chunks in {elapsed:.2f}s")
    logger.info(f"Collection: {collection_name}")


def ingest_directory(directory_path: str) -> None:
    settings = get_settings()
    logger.info(f"Ingesting directory: {directory_path}")
    logger.info(f"Chunk size: {settings.rag.chunk_size}, Overlap: {settings.rag.chunk_overlap}")

    service = create_ingestion_service()
    collection_name = settings.chroma.collection_name

    start_time = time.time()
    chunk_count = service.ingest_directory(directory_path, collection_name)
    elapsed = time.time() - start_time

    logger.info(f"Ingested {chunk_count} chunks in {elapsed:.2f}s")
    logger.info(f"Collection: {collection_name}")


def show_stats() -> None:
    settings = get_settings()
    service = create_ingestion_service()
    stats = service.get_collection_stats(settings.chroma.collection_name)

    logger.info(f"Collection: {stats['collection_name']}")
    logger.info(f"Document count: {stats['document_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to ingest")
    parser.add_argument("--directory", type=str, help="Path to directory containing PDFs")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")

    args = parser.parse_args()

    try:
        if args.stats:
            show_stats()
            return 0

        if args.pdf:
            ingest_pdf(args.pdf)
            return 0

        if args.directory:
            ingest_directory(args.directory)
            return 0

        parser.print_help()
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
